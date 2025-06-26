package llama

import (
	"container/ring"
	"encoding/binary"
	"fmt"
	"io"
	"unsafe"

	//"io"
	"math"
	"math/rand"
	"os"
	//"reflect"
	"runtime"
	"sort"
	"time"
	//"unsafe"

	"github.com/abrander/gguf"
	//progressbar "github.com/schollz/progressbar/v3"
	"github.com/mattn/go-colorable"
	"github.com/mitchellh/colorstring"
	"github.com/x448/float16"
	"golang.org/x/exp/slices"

	"github.com/gotzmann/llama.go/pkg/ml"
)

const (
	LLAMA_FILE_VERSION           = 1
	LLAMA_FILE_MAGIC             = 0x67676a74 // 'ggjt' in hex
	LLAMA_FILE_MAGIC_OLD         = 0x67676d66 // 'ggmf' in hex
	LLAMA_FILE_MAGIC_UNVERSIONED = 0x67676d6c // 'ggml' pre-versioned files
)

type ModelParams struct {
	Model  string // model path
	Prompt string

	MaxThreads int

	UseAVX  bool
	UseNEON bool

	Seed         int
	PredictCount uint32 // new tokens to predict
	RepeatLastN  uint32 // last n tokens to penalize
	PartsCount   int    // amount of model parts (-1 = determine from model dimensions)
	CtxSize      uint32 // context size
	BatchSize    uint32 // batch size for prompt processing
	KeepCount    uint32

	// --- sampling parameters

	TopK          uint32  // 40
	TopP          float32 // 0.95
	Temp          float32 // 0.80
	RepeatPenalty float32 // 1.10

	InputPrefix string   // string to prefix user inputs with
	Antiprompt  []string // string upon seeing which more user input is prompted

	MemoryFP16   bool // use f16 instead of f32 for memory kv
	RandomPrompt bool // do not randomize prompt if none provided
	UseColor     bool // use color to distinguish generations and inputs
	Interactive  bool // interactive mode

	Embedding        bool // get only sentence embedding
	InteractiveStart bool // wait for user input immediately

	Instruct   bool // instruction mode (used for Alpaca models)
	IgnoreEOS  bool // do not stop generating after eos
	Perplexity bool // compute perplexity over the prompt
	UseMLock   bool // use mlock to keep model in memory
	MemTest    bool // compute maximum memory usage

	VerbosePrompt bool
}

// pair is a C++ inspired struct
type pair struct {
	first  float32
	second uint32
}

// Context is the context of the model.
type Context struct {
	kvSelf    KVCache   // key-value store for the self attention
	Logits    []float32 // decode output 2D array [tokensCount][vocabSize]
	Embedding []float32 // input embedding 1D array [embdSize]
	MLContext *ml.Context
}

// NewContext creates a new context.
func NewContext(model *Model, params *ModelParams) *Context {
	dt := ml.TYPE_F32

	size := model.hparams.embdSize * model.hparams.layersCount * params.CtxSize
	return &Context{
		kvSelf: KVCache{
			K: ml.NewTensor1D(nil, dt, size), // Fixed OK
			V: ml.NewTensor1D(nil, dt, size), // Fixed OK
		},
		Logits:    make([]float32, model.hparams.vocabSize, model.hparams.vocabSize),
		Embedding: make([]float32, 0, 0), // FIXME: vocab.Size ?
		MLContext: ml.NewContext(params.MaxThreads, params.UseAVX, params.UseNEON),
	}
}

func (ctx *Context) ReleaseContext() {
	// not sure if it makes sense to nil explicitly
	ctx.kvSelf.K = nil
	ctx.kvSelf.V = nil
	ctx.Logits = nil
	ctx.Embedding = nil
	// close sync channel and stop compute workers
	ctx.MLContext.ReleaseContext()
}

// ContextParams are the parameters for the context.
// struct llama_context_params {
type ContextParams struct {
	CtxSize    uint32 // text context
	PartsCount int    // -1 for default
	Seed       int    // RNG seed, 0 for random
	LogitsAll  bool   // the llama_eval() call computes all logits, not just the last one
	VocabOnly  bool   // only load the vocabulary, no weights
	UseLock    bool   // force system to keep model in RAM
	Embedding  bool   // embedding mode only
}

// Layer is a single layer of the model.
type Layer struct {

	// normalization
	attentionNorm *ml.Tensor

	// attention
	wq *ml.Tensor
	wk *ml.Tensor
	wv *ml.Tensor
	wo *ml.Tensor

	// normalization
	ffn_norm *ml.Tensor

	// ff
	w1 *ml.Tensor
	w2 *ml.Tensor
	w3 *ml.Tensor
}

// HParams are the hyperparameters of the model (LLaMA-7B commented as example).
type HParams struct {
	ctxSize     uint32
	vocabSize   uint32 // 32000
	embdSize    uint32 // 4096
	multSize    uint32 // 256
	headsCount  uint32 // 32
	layersCount uint32 // 32
	rotCount    uint32 // 64
	ftype       uint32
}

// ModelType is the type of the model.
type ModelType uint8

// available llama models
const (
	MODEL_UNKNOWN ModelType = iota
	MODEL_7B
	MODEL_13B
	MODEL_30B
	MODEL_65B
)

// KVCache is a key-value cache for the self attention.
type KVCache struct {
	K *ml.Tensor
	V *ml.Tensor

	N uint32 // number of tokens currently in the cache
}

// Model is the representation of any NN model (and LLaMA too).
type Model struct {
	Type    ModelType
	ctx     *ml.Context
	hparams *HParams

	tokEmbeddings *ml.Tensor
	norm          *ml.Tensor
	output        *ml.Tensor

	layers []Layer

	tensors map[string]*ml.Tensor
}

// NewModel creates a new model with default hyperparameters.
func NewModel(params *ModelParams) *Model {
	return &Model{
		hparams: &HParams{
			ctxSize: params.CtxSize,
		},
		layers:  make([]Layer, 0),
		tensors: make(map[string]*ml.Tensor),
	}
}

// Eval runs one inference iteration over the LLaMA model
// lctx = model context with all LLaMA data
// tokens = new batch of tokens to process
// pastCount = the context size so far
// params = all other parameters like max threads allowed, etc
func Eval(
	lctx *Context,
	vocab *ml.Vocab,
	model *Model,
	tokens []uint32,
	pastCount uint32,
	params *ModelParams,
) error {

	N := uint32(len(tokens))
	kvSelf := lctx.kvSelf

	embdSize := model.hparams.embdSize
	layersCount := model.hparams.layersCount
	ctxSize := model.hparams.ctxSize
	headsCount := model.hparams.headsCount
	vocabSize := model.hparams.vocabSize
	rotCount := model.hparams.embdSize / model.hparams.headsCount

	ctx0 := lctx.MLContext

	graph := &ml.Graph{
		//MaxThreads: params.MaxThreads,
		//UseNEON:    params.UseNEON,
		//UseAVX:     params.UseAVX,
	}

	// Initialize the embd tensor with the tokensFloat32 data
	embd := ml.NewTensor1D(ctx0, ml.TYPE_I8, 0) // Data will be appended in blocks
	ml.HydrateTensorFromUI32(embd, tokens)

	inpL := ml.GetRows(ctx0, model.tokEmbeddings, embd)

	for il := uint32(0); il < layersCount; il++ {

		//if il > 0 {
		//	break // DEBUG
		//}

		inpSA := inpL

		// norm
		cur := ml.RMSNorm(ctx0, inpL)

		// cur = attention_norm*cur
		rep := ml.Repeat(ctx0, model.layers[il].attentionNorm, cur)
		cur = ml.Mul(ctx0, rep, cur)

		// self-attention
		{
			Qcur := ml.MulMat(ctx0, model.layers[il].wq, cur)
			Kcur := ml.MulMat(ctx0, model.layers[il].wk, cur)
			Vcur := ml.MulMat(ctx0, model.layers[il].wv, cur)

			// store key and value to memory
			if N >= 1 {

				////struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, N*n_embd, (ggml_element_size(kv_self.k)*n_embd)*(il*n_ctx + n_past));
				////struct ggml_tensor * v = ggml_view_1d(ctx0, kv_self.v, N*n_embd, (ggml_element_size(kv_self.v)*n_embd)*(il*n_ctx + n_past));

				// NB! ggml_element_size(kv_self.k) = 2 for FP16
				k := ml.View1D(ctx0, kvSelf.K, N*embdSize, embdSize*(il*ctxSize+pastCount))
				v := ml.View1D(ctx0, kvSelf.V, N*embdSize, embdSize*(il*ctxSize+pastCount))

				ml.BuildForwardExpand(graph, ml.Copy(ctx0, Kcur, k))
				ml.BuildForwardExpand(graph, ml.Copy(ctx0, Vcur, v))
			}

			Q :=
				ml.Permute(ctx0,
					ml.Rope(ctx0,
						ml.Copy(ctx0,
							Qcur,
							ml.NewTensor3D(ctx0, ml.TYPE_F32, embdSize/headsCount, headsCount, N)), // Reusable OK
						pastCount, rotCount, 0),
					0, 2, 1, 3)

			K :=
				ml.Permute(ctx0,
					ml.Rope(ctx0,
						ml.Reshape3D(ctx0,
							ml.View1D(ctx0, kvSelf.K, (pastCount+N)*embdSize, il*ctxSize*embdSize),
							embdSize/headsCount, headsCount, pastCount+N),
						pastCount, rotCount, 1),
					0, 2, 1, 3)

			// K * Q
			KQ := ml.MulMat(ctx0, K, Q)

			//@todo
			// KQ_scaled = KQ / sqrt(n_embd/n_head)
			KQScaled :=
				ml.Scale(ctx0,
					KQ,
					ml.NewI8(ctx0, float32(1.0/math.Sqrt(float64(embdSize)/float64(headsCount))), 1),
				)

			// KQ_masked = mask_past(KQ_scaled)
			KQMasked := ml.DiagMaskInf(ctx0, KQScaled, pastCount)

			// KQ = soft_max(KQ_masked)
			KQSoftMax := ml.SoftMax(ctx0, KQMasked)

			VTrans :=
				ml.Copy(ctx0,
					ml.Permute(ctx0,
						ml.Reshape3D(ctx0,
							ml.View1D(ctx0, kvSelf.V, (pastCount+N)*embdSize, il*ctxSize*embdSize),
							embdSize/headsCount, headsCount, pastCount+N),
						1, 2, 0, 3),
					ml.NewTensor3D(ctx0, ml.TYPE_F32 /* kv_self.v->type */, pastCount+N, embdSize/headsCount, headsCount))

			// KQV = transpose(V) * KQ_soft_max
			KQV := ml.MulMat(ctx0, VTrans, KQSoftMax)

			// KQV_merged = KQV.permute(0, 2, 1, 3)
			KQVMerged := ml.Permute(ctx0, KQV, 0, 2, 1, 3)

			// cur = KQV_merged.contiguous().view(n_embd, N)
			cur = ml.Copy(ctx0,
				KQVMerged,
				ml.NewTensor2D(ctx0, ml.TYPE_F32, embdSize, N)) // Reusable OK

			// projection (no bias)
			cur = ml.MulMat(ctx0, model.layers[il].wo, cur)

		}

		inpFF := ml.Add(ctx0, cur, inpSA)

		// feed-forward network
		{
			// norm
			{
				cur = ml.RMSNorm(ctx0, inpFF)

				// cur = ffn_norm*cur
				cur = ml.Mul(ctx0,
					ml.Repeat(ctx0, model.layers[il].ffn_norm, cur),
					cur)
			}

			tmp := ml.MulMat(ctx0, model.layers[il].w3, cur)

			cur = ml.MulMat(ctx0, model.layers[il].w1, cur)

			// SILU activation
			cur = ml.Silu(ctx0, cur)

			cur = ml.Mul(ctx0, cur, tmp)

			cur = ml.MulMat(ctx0, model.layers[il].w2, cur)
		}

		cur = ml.Add(ctx0, cur, inpFF)

		// input for next layer
		inpL = cur
	}

	// --- norm

	inpL = ml.RMSNorm(ctx0, inpL)

	// inpL = norm*inpL
	inpL = ml.Mul(ctx0,
		ml.Repeat(ctx0, model.norm, inpL),
		inpL)

	embeddings := inpL

	// lm_head
	inpL = ml.MulMat(ctx0, model.output, inpL)

	// run the computation
	ml.BuildForwardExpand(graph, inpL)

	ml.GraphCompute(ctx0, graph)

	// --- extract logits

	// Copy only the relevant part of inpL.Data to lctx.Logits
	for i := uint32(0); i < vocabSize; i++ {
		srcIndex := vocabSize*(N-1) + i
		if i >= uint32(len(lctx.Logits)) || srcIndex >= uint32(len(inpL.Data)) {
			fmt.Println("Error: Index out of bounds during Logits copy")
			os.Exit(1)
		}
		lctx.Logits[i] = float32(inpL.Scalars[srcIndex/32]) * float32(inpL.Data[srcIndex])
	}

	if ml.DEBUG {
		printTensor(inpL, "INPL")

		fmt.Printf("\n\n=== LOGITS === %d ===\n", len(lctx.Logits)) // DEBUG
		for ii := 0; ii < 13; ii++ {
			fmt.Printf("%.4f  ", lctx.Logits[ii])
		}
	}

	// --- extract embeddings

	if len(lctx.Embedding) > 0 {
		////memcpy(embedding_out.data(), (float *) ggml_get_data(embeddings) + (n_embd*(N - 1)), sizeof(float)*n_embd);
		for i := uint32(0); i < embdSize; i++ {
			index := (embdSize * (N - 1)) + i
			lctx.Embedding[i] = float32(embeddings.Scalars[index/32]) * float32(embeddings.Data[index])
		}
	}

	// It really helps to eliminate degradation of performance when
	// the garbage collector do it job more often
	runtime.GC()

	return nil
}

// printTensor prints a tensor
func printTensor(tensor *ml.Tensor, name string) {
	var dt string
	if tensor.Type == ml.TYPE_F16 {
		dt = "FP16"
	}
	if tensor.Type == ml.TYPE_F32 {
		dt = "FP32"
	}
	if tensor.Type == ml.TYPE_Q4_0 {
		dt = "INT4"
	}

	fmt.Printf("\n\n=== [ %s | %s | %d:%d:%d ] ===\n",
		name, dt, tensor.NE[0], tensor.NE[1], tensor.NE[2])

	for nn := 0; nn < min(12, int(tensor.NE[1])); nn++ {
		fmt.Printf("\n %d x %d ...\t", nn, tensor.NE[0])
		for ii := 0; ii < min(12, int(tensor.NE[0])); ii++ {
			fmt.Printf("%.3f\t", tensor.Data[nn*int(tensor.NE[0])+ii])
		}
	}
}

// SampleTopPTopK samples next token given probabilities for each embedding:
//   - consider only the top K tokens
//   - from them, consider only the top tokens with cumulative probability > P
func SampleTopPTopK(
	logits []float32,
	lastNTokens *ring.Ring, // TODO: Use custom performant container
	lastNTokensSize uint32, // TODO: Remove
	topK uint32,
	topP float32,
	temp float32,
	repeatPenalty float32,
) uint32 {

	logitsCount := uint32(len(logits))

	if ml.DEBUG {
		fmt.Printf("\n\n>>> SampleTopPTopK <<<\n")
		fmt.Printf("\n=== LOGITS | %d ===\n", len(logits))
		for i := 0; i < 8; i++ {
			fmt.Printf("%.4f ", logits[i])
		}
		fmt.Printf(" ... ")
		for i := int(len(logits)) - 1; i >= int(len(logits))-8; i-- {
			fmt.Printf("%.4f ", logits[i])
		}
		extractedTokens := ExtractTokens(lastNTokens.Move(-int(lastNTokensSize)), int(lastNTokensSize))
		fmt.Printf("\n=== LAST N TOKENS | %d ===\n", len(extractedTokens))
		for i := 0; i < int(lastNTokensSize); i++ {
			fmt.Printf("%d ", extractedTokens[i])
		}
	}

	////if (temp <= 0) {
	////    // select the token with the highest logit directly
	////    float max_logit = plogits[0];
	////    llama_vocab::id max_id = 0;
	////
	////    for (int i = 1; i < n_logits; ++i) {
	////        if (plogits[i] > max_logit) {
	////            max_logit = plogits[i];
	////            max_id = i;
	////        }
	////    }
	////    return max_id;
	////}

	logitsID := make([]pair, 0, logitsCount)

	scale := float32(1.0 / temp)
	for i := uint32(0); i < logitsCount; i++ {

		// Repetition penalty from ctrl paper (https://arxiv.org/abs/1909.05858)
		// Credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main

		// Check if the i-th token is present in the last_n_tokens ring buffer
		tokenExists := false
		// TODO: Ompimize [ 32,000 * 1024 ~ 100 ms ] loop with better data structure for lastNTokens
		lastNTokens.Do(func(p interface{}) {
			if p.(uint32) == i {
				tokenExists = true
			}
		})

		// If lastNTokens already contains i-th token, append it with repeat penalty
		if tokenExists {
			// If score < 0, then repetition penalty has to be multiplied to reduce the previous token probability
			if logits[i] < 0.0 {
				logitsID = append(logitsID, pair{logits[i] * scale * repeatPenalty, i})
			} else {
				logitsID = append(logitsID, pair{logits[i] * scale / repeatPenalty, i})
			}
			// Else append pair to logitsID, scaling probability
		} else {
			logitsID = append(logitsID, pair{logits[i] * scale, i})
		}
	}

	if ml.DEBUG {
		fmt.Printf("\n=== LOGITS ID AFTER | %d ===\n", len(logitsID))
		for i := 0; i < min(6, len(logitsID)); i++ {
			fmt.Printf("{ %.3f | %d }", logitsID[i].first, logitsID[i].second)
		}
		fmt.Printf(" ... ")
		for i := len(logitsID) - 6; i < len(logitsID)-1; i++ {
			fmt.Printf("{ %.3f | %d } ", logitsID[i].first, logitsID[i].second)
		}
	}

	// --- sort logitsID slice and return only top K elements

	// std::partial_sort
	// Rearranges elements such that the range [first, middle) contains
	// the sorted middle − first smallest elements in the range [first, last).
	// The order of equal elements is not guaranteed to be preserved.
	// The order of the remaining elements in the range [middle, last) is unspecified.

	sort.Slice(
		logitsID, // logitsID[:topK],
		func(a, b int) bool {
			return logitsID[a].first > logitsID[b].first
		})

	if ml.DEBUG {
		fmt.Printf("\n=== LOGITS ID SORTED | TOP K = %d ===\n", topK)
		for i := 0; i < min(6, len(logitsID)); i++ {
			fmt.Printf("{ %.3f | %d }", logitsID[i].first, logitsID[i].second)
		}
		fmt.Printf(" ... ")
		for i := len(logitsID) - 6; i < len(logitsID)-1; i++ {
			fmt.Printf("{ %.3f | %d } ", logitsID[i].first, logitsID[i].second)
		}
	}

	logitsID = logitsID[:topK]

	if ml.DEBUG {
		fmt.Printf("\n=== LOGITS ID RESIZED | %d ===\n", len(logitsID))
		for i := 0; i < min(6, len(logitsID)); i++ {
			fmt.Printf("{ %.3f | %d }", logitsID[i].first, logitsID[i].second)
		}
		fmt.Printf(" ... ")
		for i := len(logitsID) - 6; i < len(logitsID)-1; i++ {
			fmt.Printf("{ %.3f | %d } ", logitsID[i].first, logitsID[i].second)
		}
	}

	// Since logitsID is already sorted, the max value is the first element
	maxl := logitsID[0].first

	// Compute probabilities for the top k tokens
	probs := make([]float32, len(logitsID))

	sum := 0.0
	for i, kv := range logitsID {
		p := math.Exp(float64(kv.first - maxl))
		probs[i] = float32(p)
		sum += p
	}

	if ml.DEBUG {
		fmt.Printf("\n=== PROBS | %d ===\n", len(probs))
		for i := 0; i < min(6, len(probs)); i++ {
			fmt.Printf("%.3f  ", probs[i])
		}
		fmt.Printf(" ... ")
		for i := len(logitsID) - 6; i < len(probs)-1; i++ {
			fmt.Printf("%.3f  ", probs[i])
		}
	}

	// normalize the probs
	for i := range probs {
		probs[i] /= float32(sum)
	}

	if ml.DEBUG {
		fmt.Printf("\n=== PROBS NORM | %d ===\n", len(probs))
		for i := 0; i < min(6, len(probs)); i++ {
			fmt.Printf("%.3f  ", probs[i])
		}
		fmt.Printf(" ... ")
		for i := len(logitsID) - 6; i < len(probs)-1; i++ {
			fmt.Printf("%.3f  ", probs[i])
		}
	}

	if topP < 1.0 {

		cumsum := float32(0.0) // TODO float64 for better math?
		for i := uint32(0); i < uint32(len(probs)); i++ {
			cumsum += probs[i]
			if cumsum >= topP {
				probs = probs[:i+1]
				logitsID = logitsID[:i+1]
				break
			}
		}

		cumsum = 1.0 / cumsum
		for i := uint32(0); i < uint32(len(probs)); i++ {
			probs[i] *= cumsum
		}
	}

	if ml.DEBUG {
		if len(probs) > 6 {
			fmt.Printf("\n=== PROBS POST | %d ===\n", len(probs))
			for i := 0; i < min(6, len(probs)); i++ {
				fmt.Printf("%.3f  ", probs[i])
			}
			fmt.Printf(" ... ")
			for i := len(logitsID) - 6; i < len(probs)-1; i++ {
				fmt.Printf("%.3f  ", probs[i])
			}
		}
	}

	// --- Hand-crafted Discrete Distribution math - do we need something better?

	// Original C++ version with rng = std::mt19937
	// Mersenne Twister pseudo-random generator of 32-bit numbers with a state size of 19937 bits.

	// std::discrete_distribution<> dist(probs.begin(), probs.end());
	// int idx = dist(rng);
	// return logits_id[idx].second;

	seed := time.Now().UnixNano()
	source := rand.NewSource(seed)

	for i := 0; i < len(probs); i++ {
		f := float32(source.Int63()) / (1 << 63)
		probs[i] = probs[i] * probs[i] * f * f
	}

	idx := 0
	maxProb := probs[0]
	for i := 1; i < len(probs); i++ {
		if probs[i] > maxProb {
			idx = i
			maxProb = probs[i]
		}
	}

	if ml.DEBUG {
		fmt.Printf("\n=== PROVED === ")
		for i := 0; i < min(8, len(probs)); i++ {
			fmt.Printf("%.3f | ", probs[i])
		}
		fmt.Printf(" === idx = %d | logitsID = %d | weight = %.3f | ", idx, logitsID[idx].second, logitsID[idx].first)
	}

	/*
		// --- experimental approach seems doesn't work right yet

		rng := rand.New(source)

		cumulative := make([]float32, len(probs))
		cumulative[0] = probs[0]
		for i := 1; i < len(probs); i++ {
			cumulative[i] = cumulative[i-1] + probs[i]
		}

		target := rng.Float32() * cumulative[len(cumulative)-1]
		idx := sort.Search(len(cumulative), func(i int) bool { return cumulative[i] >= target })

		if ml.DEBUG {
			fmt.Printf("\n=== EXPERIMENTAL === ")
			for i := 0; i < min(8, len(probs)); i++ {
				fmt.Printf("%.3f | ", probs[i])
			}
			fmt.Printf(" === idx = %d | logitsID = %d | weight = %.3f | ", idx, logitsID[idx].second, logitsID[idx].first)
		}
	*/

	return logitsID[idx].second
}

// LoadModel loads a model's weights from a file
// See convert-pth-to-ggml.py for details on format
// func LoadModel(fileName string, params ModelParams, silent bool) (*Context, error) {
func LoadModel(fileName string, params *ModelParams, silent bool) (*ml.Vocab, *Model, error) {

	g, err := gguf.OpenFile(fileName)

	if err != nil {
		return nil, nil, err
	}

	fmt.Printf("Got GGUF file version: %d\n", g.Version)

	arch, _ := g.Metadata.String("general.architecture")
	fmt.Printf(arch)

	contextLength, _ := g.Metadata.Int("llama.context_length")
	fmt.Printf("Context length: %d\n", contextLength)

	// --- load hparams

	model := NewModel(params)

	// Populate hparams from the metadata map using standard GGUF keys.
	model.hparams.vocabSize = g.Metadata["llama.vocab_size"].(uint32)
	model.hparams.embdSize = g.Metadata["llama.embedding_length"].(uint32)
	model.hparams.headsCount = g.Metadata["llama.attention.head_count"].(uint32)
	model.hparams.layersCount = g.Metadata["llama.block_count"].(uint32)
	model.hparams.rotCount = g.Metadata["llama.rope.dimension_count"].(uint32)
	// GGUF doesn't have a direct 'ftype' in the same way; it's per-tensor.
	// You'll determine the type when you read each tensor's info.
	// For a general model ftype, you might check the type of a major tensor.
	model.hparams.ftype = 4 // Default or determine from tensor info later

	// The 'multiple_of' parameter is part of llama.feed_forward_length calculation in GGUF.
	// We'll use the pre-calculated feed_forward_length directly.
	ffSize := g.Metadata["llama.feed_forward_length"].(uint32)

	// In GGUF, context size is also in the metadata.
	model.hparams.ctxSize = g.Metadata["llama.context_length"].(uint32)

	multSize := 256

	vocab := ml.NewVocab(model.hparams.vocabSize)

	if ml.DEBUG {
		fmt.Printf("\nvocab  = %d", model.hparams.vocabSize)
		fmt.Printf("\nembd   = %d", model.hparams.embdSize)
		fmt.Printf("\nmult   = %d", multSize)
		fmt.Printf("\nheads  = %d", model.hparams.headsCount)
		fmt.Printf("\nlayers = %d", model.hparams.layersCount)
		fmt.Printf("\nff     = %d", ffSize)
		fmt.Printf("\nrot    = %d", model.hparams.rotCount)
		fmt.Printf("\nftype    = %d", model.hparams.ftype)
	}

	// --- load vocab

	if !silent && runtime.GOOS == "windows" {
		Colorize("[magenta][ INIT ][white] Loading vocab...")
	}

	tokensList, err := gguf.MetaValue[[]string](g.Metadata, "tokenizer.ggml.tokens")
	if err != nil {
		fmt.Printf("could not read tokens: %w", err)
		return nil, nil, fmt.Errorf("could not read tokens: %w", err)
	}

	scoresList, err := gguf.MetaValue[[]float32](g.Metadata, "tokenizer.ggml.scores")
	if err != nil {
		fmt.Printf("could not read scores: %w", err)
		return nil, nil, fmt.Errorf("could not read scores: %w", err)
	}

	if len(tokensList) != len(scoresList) {
		fmt.Printf("token count (%d) does not match score count (%d)", len(tokensList), len(scoresList))
		return nil, nil, fmt.Errorf("token count (%d) does not match score count (%d)", len(tokensList), len(scoresList))
	}

	// Fill vocab
	for i := 0; i < len(tokensList); i++ {
		token := tokensList[i]
		score := scoresList[i]

		vocab.Token2ID[token] = uint32(i)
		vocab.ID2Token[uint32(i)] = ml.TokenScore{Token: token, Score: score}
	}

	// --- prepare memory for the weights
	{
		model.tokEmbeddings = ml.NewTensor2D(nil, ml.DType(model.hparams.ftype) /*wtype*/, model.hparams.embdSize, model.hparams.vocabSize) // Fixed OK

		model.norm = ml.NewTensor1D(nil, ml.DType(model.hparams.ftype), model.hparams.embdSize)                                      // Fixed OK
		model.output = ml.NewTensor2D(nil, ml.DType(model.hparams.ftype) /*wtype*/, model.hparams.embdSize, model.hparams.vocabSize) // Fixed OK

		// map by name
		model.tensors["token_embd.weight"] = model.tokEmbeddings

		model.tensors["output_norm.weight"] = model.norm
		model.tensors["output.weight"] = model.output

		model.layers = make([]Layer, model.hparams.layersCount)
		for i := uint32(0); i < model.hparams.layersCount; i++ {

			model.layers[i].attentionNorm = ml.NewTensor1D(nil, ml.DType(model.hparams.ftype), model.hparams.embdSize) // Fixed OK

			model.layers[i].wq = ml.NewTensor2D(nil, ml.DType(model.hparams.ftype) /*wtype*/, model.hparams.embdSize, model.hparams.embdSize) // Fixed OK
			model.layers[i].wk = ml.NewTensor2D(nil, ml.DType(model.hparams.ftype) /*wtype*/, model.hparams.embdSize, model.hparams.embdSize) // Fixed OK
			model.layers[i].wv = ml.NewTensor2D(nil, ml.DType(model.hparams.ftype) /*wtype*/, model.hparams.embdSize, model.hparams.embdSize) // Fixed OK
			model.layers[i].wo = ml.NewTensor2D(nil, ml.DType(model.hparams.ftype) /*wtype*/, model.hparams.embdSize, model.hparams.embdSize) // Fixed OK

			model.layers[i].ffn_norm = ml.NewTensor1D(nil, ml.DType(model.hparams.ftype), model.hparams.embdSize)

			model.layers[i].w1 = ml.NewTensor2D(nil, ml.DType(model.hparams.ftype) /*wtype*/, model.hparams.embdSize, ffSize) // Fixed OK
			model.layers[i].w2 = ml.NewTensor2D(nil, ml.DType(model.hparams.ftype) /*wtype*/, ffSize, model.hparams.embdSize) // Fixed OK
			model.layers[i].w3 = ml.NewTensor2D(nil, ml.DType(model.hparams.ftype) /*wtype*/, model.hparams.embdSize, ffSize) // Fixed OK

			// map by name
			prefix := fmt.Sprintf("blk.%d.", i)

			model.tensors[prefix+"attn_norm.weight"] = model.layers[i].attentionNorm

			model.tensors[prefix+"attn_q.weight"] = model.layers[i].wq
			model.tensors[prefix+"attn_k.weight"] = model.layers[i].wk
			model.tensors[prefix+"attn_v.weight"] = model.layers[i].wv
			model.tensors[prefix+"attn_output.weight"] = model.layers[i].wo

			model.tensors[prefix+"ffn_norm.weight"] = model.layers[i].ffn_norm

			model.tensors[prefix+"ffn_gate.weight"] = model.layers[i].w1
			model.tensors[prefix+"ffn_up.weight"] = model.layers[i].w2
			model.tensors[prefix+"ffn_down.weight"] = model.layers[i].w3

		}
	}

	if !silent /* && runtime.GOOS == "windows" */ {
		//Colorize("[magenta][ INIT ][white] Loading model - please wait ...")
		Colorize("[light_magenta][ INIT ][light_blue] Loading model, please wait ")
	}

	for i, tInfo := range g.Tensors {
		dims := len(tInfo.Dimensions)
		if dims < 1 || dims > 2 { // TODO Check for EOF
			break
		}

		shardType := mapGGUFTypeToDType(tInfo.Type)

		nelements := 1
		ne := [2]uint32{1, 1}
		for i := 0; i < dims; i++ {
			ne[i] = uint32(tInfo.Dimensions[i])
			nelements *= int(ne[i])
		}

		name := tInfo.Name

		tensor, ok := model.tensors[name]
		if !ok {
			fmt.Printf("\n[ERROR] Unknown tensor '%s' in model file", name)
			os.Exit(1)
		}

		tensorSize := tensor.Nelements()

		switch shardType {
		case ml.TYPE_I8:
			r, _ := tInfo.Reader()

			blockSize := uint32(32)
			blocksCount := tensorSize / blockSize
			blockBytes := uint32(2 + 32)

			buf := make([]byte, blockBytes*blocksCount)
			if _, err := io.ReadFull(r, buf); err != nil {
				fmt.Printf("\n[ERROR] Error while reading tensor %s", err.Error())

				return nil, nil, err
			}

			for i := uint32(0); i < blocksCount; i++ {
				offset := i * blockBytes
				scalarBytes := buf[offset : offset+2]
				weightBytes := buf[offset+2 : offset+34]

				// Read scalar
				scalar := binary.LittleEndian.Uint16(scalarBytes)
				tensor.Scalars[i] = *(*float16.Float16)(unsafe.Pointer(&scalar))

				// Read weights
				for j := uint32(0); j < blockSize; j++ {
					tensor.Data[i*blockSize+j] = int8(weightBytes[j])
				}
			}

		default:
			fmt.Printf("\n[ERROR] Tensor data type is not supported yet!")
			os.Exit(0)
		}

		// TODO: Implement just simple dots increasing count for Windows
		if !silent && i%10 == 0 {
			Colorize("[light_blue].")
		}
	}

	return vocab, model, nil
}

// mapGGUFTypeToDType converts a GGUF tensor type to the internal ml.DType.
// This is a placeholder and depends on your gguf library's type definitions.
func mapGGUFTypeToDType(ggufType gguf.GGML) ml.DType {
	switch ggufType {
	case gguf.GgmlQ8_0:
		return ml.TYPE_I8
	default:
		return ml.TYPE_I8 // or handle as an error
	}
}

// max returns the maximum of two float32 values
func max(a, b float32) float32 {
	if a >= b {
		return a
	}
	return b
}

// readInt reads 32-bit integer from the file
func readInt(file *os.File) uint32 {
	buf := make([]byte, 4)
	if count, err := file.Read(buf); err != nil || count != 4 {
		return 0
	}
	return uint32(buf[3])<<24 | uint32(buf[2])<<16 | uint32(buf[1])<<8 | uint32(buf[0])
}

func readInt8(reader io.Reader) int8 {
	buf := make([]byte, 1)
	if count, err := reader.Read(buf); err != nil || count != 1 {
		return 0
	}
	return int8(buf[0])
}

// readString reads a string from the file
func readString(file *os.File, len uint32) string {
	buf := make([]byte, len)
	if count, err := file.Read(buf); err != nil || count != int(len) {
		return ""
	}
	return string(buf)
}

// readFP16ToFP32 reads a 16-bit float from the file and converts it to 32-bit
func readFP16ToFP32(file *os.File) float32 {
	buf := make([]byte, 2)
	if count, err := file.Read(buf); err != nil || count != 2 {
		return 0.0
	}
	bits := uint16(buf[1])<<8 | uint16(buf[0])
	f16 := float16.Frombits(bits)
	return f16.Float32()
}

// readFP32 reads a 32-bit float from the file
func readFP32(reader io.Reader) float32 {
	buf := make([]byte, 4)
	if count, err := reader.Read(buf); err != nil || count != 4 {
		return 0.0
	}
	bits := uint32(buf[3])<<24 | uint32(buf[2])<<16 | uint32(buf[1])<<8 | uint32(buf[0])
	return math.Float32frombits(bits)
}

// ExtractTokens is a function to extract a slice of tokens from the ring buffer
func ExtractTokens(r *ring.Ring, count int) []uint32 {
	tokens := make([]uint32, count)
	for i := 0; i < count; i++ {
		tokens[i] = r.Value.(uint32)
		r = r.Next()
	}
	return tokens
}

// Colorize is a function to print colored text to the console
func Colorize(format string, opts ...interface{}) (n int, err error) {
	var DefaultOutput = colorable.NewColorableStdout()
	return fmt.Fprintf(DefaultOutput, colorstring.Color(format), opts...)
}

// min returns the minimum of a and b.
func min(a, b int) int {
	if a <= b {
		return a
	}
	return b
}

// Resize() (safe) for using instead of C++ std::vector:resize()
// https://go.dev/play/p/VlQ7N75E5AD
func Resize(slice []float32, size int) []float32 {
	newSlice := make([]float32, size)
	for i := 0; i < min(size, len(slice)); i++ {
		newSlice[i] = slice[i]
	}
	return newSlice
}

// NB! This do not clear the underlying array when resizing
// https://go.dev/play/p/DbK4dFqwrZn
func ResizeInplace(slice *[]float32, size int) {
	if len(*slice) == size {
		return
	} else if size < len(*slice) {
		*slice = (*slice)[:size]
	} else {
		*slice = slices.Grow(*slice, size)
		*slice = (*slice)[:size]
	}
}
