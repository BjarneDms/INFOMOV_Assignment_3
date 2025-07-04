package ml

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"github.com/x448/float16"
)

const (
	DEBUG = false

	MAX_DIMS   = 4
	MAX_NODES  = 4096
	MAX_PARAMS = 16
	MAX_OPT    = 4

	QK = 32 // quantization

	TOKEN_BOS = 1
	TOKEN_EOS = 2
)

// computation graph
type Graph struct {
	//MaxThreads int

	//UseAVX  bool
	//UseNEON bool

	NodesCount uint32
	LeafsCount uint32

	Jobs chan *ComputeParams

	Nodes [MAX_NODES]*Tensor
	Grads [MAX_NODES]*Tensor
	Leafs [MAX_NODES]*Tensor
}

type InitParams struct {
}

type Context struct {
	MaxThreads int
	UseAVX     bool
	UseNEON    bool
	//Graph      *Graph
	Compute   chan *ComputeParams
	Allocator *Allocator
}

func NewContext(maxThreads int, useAVX, useNEON bool) *Context {

	ch := make(chan *ComputeParams, maxThreads) // TODO: +1 for safety?

	for i := 0; i < maxThreads; i++ {
		go Job(ch, i)
	}

	return &Context{
		MaxThreads: maxThreads,
		UseAVX:     useAVX,
		UseNEON:    useNEON,
		Compute:    ch,
		Allocator:  NewAllocator(),
	}
}

// ReleaseContext frees all context resources - channel will be closed and goroutines stopped
func (ctx *Context) ReleaseContext() {
	close(ctx.Compute)
	// TODO: Maybe some steps for Allocator too
}

type DType uint8

// Data types are the same as in llama.cpp so full compatibility there
const (
	TYPE_F32   DType = 0
	TYPE_F16   DType = 1
	TYPE_Q4_0  DType = 2
	TYPE_Q4_1  DType = 3
	TYPE_I8    DType = 4
	TYPE_I16   DType = 5
	TYPE_I32   DType = 6
	TYPE_COUNT DType = 8
)

func printTensor(tensor *Tensor, name string) {

	var dt string
	switch tensor.Type {
	case TYPE_F16:
		dt = "FP16"
	case TYPE_F32:
		dt = "FP32"
	case TYPE_Q4_0:
		dt = "INT4"
	case TYPE_I8:
		dt = "INT8"
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

// precomputed exp table for f16 (128 KB)
// static ggml_fp16_t table_exp_f16[1 << 16];
var TableExpFP16 [1 << 16]float16.Float16

var BLCK_SIZE [TYPE_COUNT]uint32 = [TYPE_COUNT]uint32{1, 1, QK, QK, 1, 1, 1, 0}
var TYPE_SIZE [TYPE_COUNT]uint32 = [TYPE_COUNT]uint32{4, 2, 4 + QK/2, 4*2 + QK/2, 1, 2, 4, 0}

func TypeSizeFloat(dt DType) float32 {
	return float32(TYPE_SIZE[dt]) / float32(BLCK_SIZE[dt])
}

// available tensor operations
type optype uint8

const (
	OP_NONE optype = iota
	OP_DUP
	OP_ADD
	OP_SUB
	OP_MUL
	OP_DIV
	OP_SQR
	OP_SQRT
	OP_SUM
	OP_MEAN
	OP_REPEAT
	OP_ABS
	OP_SGN
	OP_NEG
	OP_STEP
	OP_RELU
	OP_GELU
	OP_SILU
	OP_NORM
	OP_RMS_NORM

	OP_MUL_MAT

	OP_SCALE
	OP_CPY
	OP_RESHAPE
	OP_VIEW
	OP_PERMUTE
	OP_TRANSPOSE
	OP_GET_ROWS
	OP_DIAG_MASK_INF
	OP_SOFT_MAX
	OP_ROPE
	OP_CONV_1D_1S
	OP_CONV_1D_2S

	OP_FLASH_ATTN
	OP_FLASH_FF

	OP_COUNT
)

// Tensor of up to 4x dimensions
// The multi-dimensional tensors are stored in row-major order
// and the array indexes are written row-first (lexicographical access order)

type Tensor struct {
	Type DType

	Reusable bool // this tensor Data buffer might be reused with pooling

	Dims uint32

	NE [MAX_DIMS]uint32 // number of elements
	NB [MAX_DIMS]uint32 // stride in bytes

	op optype

	isParam bool

	grad *Tensor
	src0 *Tensor
	src1 *Tensor

	opt [MAX_OPT]*Tensor // FIXME: Do we need this?

	TasksCount int

	Scalars []float16.Float16
	Data    []int8
}

// ggml_is_contiguous
func (tensor *Tensor) IsContiguous() bool {
	return tensor.NB[0] == TYPE_SIZE[tensor.Type] &&
		tensor.NB[1] == tensor.NB[0]*tensor.NE[0]/BLCK_SIZE[tensor.Type] &&
		tensor.NB[2] == tensor.NB[1]*tensor.NE[1] &&
		tensor.NB[3] == tensor.NB[2]*tensor.NE[2]
}

func AreSameShape(a, b *Tensor) bool {
	return (a.NE[0] == b.NE[0]) && (a.NE[1] == b.NE[1]) && (a.NE[2] == b.NE[2]) && (a.NE[3] == b.NE[3])
}

func (t *Tensor) Nelements() uint32 {
	return t.NE[0] * t.NE[1] * t.NE[2] * t.NE[3]
}

func (t *Tensor) Nrows() uint32 {
	return t.NE[1] * t.NE[2] * t.NE[3]
}

// ggml_nbytes
func (t *Tensor) Nbytes() uint32 {
	return (t.Nelements() * TYPE_SIZE[t.Type]) / BLCK_SIZE[t.Type]
}

// ggml_view_tensor
func ViewTensor(ctx *Context, src *Tensor) *Tensor {
	return NewTensor(ctx, src.Type, src.Dims, src.NE[0], src.NE[1], src.NE[2], src.NE[3], src.Scalars, src.Data)
}

// ggml_dup_tensor
func DupTensor(ctx *Context, src *Tensor) *Tensor {
	return NewTensor(ctx, src.Type, src.Dims, src.NE[0], src.NE[1], src.NE[2], src.NE[3], nil, nil) // Reusbale OK
}

// struct ggml_tensor * Mul(
func Mul(ctx *Context, a, b *Tensor) *Tensor {
	return MulImpl(ctx, a, b, false)
}

// struct ggml_tensor * Mul_inplace(
func MulInplace(ctx *Context, a, b *Tensor) *Tensor {
	return MulImpl(ctx, a, b, true)
}

// struct ggml_tensor * Mul_impl(
func MulImpl(ctx *Context, a, b *Tensor, inplace bool) *Tensor {
	////ASSERT(ggml_are_same_shape(a, b));

	if !AreSameShape(a, b) {
		fmt.Printf("\n[STOP] MulImpl - tensors of different shapes!")
		os.Exit(1)
	}

	isNode := false

	if inplace && (a.grad != nil || b.grad != nil) {
		isNode = true
	}

	if inplace {
		////ASSERT(is_node == false);
	}

	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_MUL
	result.src0 = a
	result.src1 = b

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

// ggml_can_mul_mat
func CanMulMat(t0, t1 *Tensor) bool {
	return (t0.NE[0] == t1.NE[0]) && (t0.NE[2] == t1.NE[2]) && (t0.NE[3] == t1.NE[3])
}

// ggml_mul_mat
func MulMat(ctx *Context, a, b *Tensor) *Tensor {
	////ASSERT(ggml_can_mul_mat(a, b));
	////GGML_ASSERT(!ggml_is_transposed(a));

	isNode := false

	if a.grad != nil || b.grad != nil {
		isNode = true
	}

	result := NewTensor(ctx, TYPE_I8, min32(a.Dims, b.Dims), a.NE[1], b.NE[1], a.NE[2], b.NE[3], nil, nil) // Reusable OK

	result.op = OP_MUL_MAT
	result.src0 = a
	result.src1 = b

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

// ggml_add
func AddImpl(ctx *Context, a, b *Tensor, inplace bool) *Tensor {
	////ASSERT(ggml_are_same_shape(a, b));

	//bool is_node = false;

	////if (!inplace && (a.grad || b.grad)) {
	////is_node = true;
	////}

	////struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_ADD
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = b

	return result
}

func Add(ctx *Context, a, b *Tensor) *Tensor {
	return AddImpl(ctx, a, b, false)
}

func AddInplace(ctx *Context, a, b *Tensor) *Tensor {
	return AddImpl(ctx, a, b, true)
}

// ggml_sum
func Sum(ctx *Context, a *Tensor) *Tensor {
	isNode := false

	if a.grad != nil {
		isNode = true
	}

	result := NewTensor1D(ctx, a.Type, 1) // Reusable OK

	result.op = OP_SUM
	result.src0 = a
	result.src1 = nil

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

// ggml_sub
func SubImpl(ctx *Context, a, b *Tensor, inplace bool) *Tensor {
	////ASSERT(ggml_are_same_shape(a, b));

	////bool is_node = false;

	////if (!inplace && (a.grad || b.grad)) {
	////is_node = true;
	////}

	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_SUB
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = b

	return result
}

func Sub(ctx *Context, a, b *Tensor) *Tensor {
	return SubImpl(ctx, a, b, false)
}

func SubInplace(ctx *Context, a, b *Tensor) *Tensor {
	return SubImpl(ctx, a, b, true)
}

// ggml_div
func DivImpl(ctx *Context, a, b *Tensor, inplace bool) *Tensor {
	////ASSERT(ggml_are_same_shape(a, b));

	////bool is_node = false;

	////if (!inplace && (a->grad || b->grad)) {
	////is_node = true;
	////}

	////if (inplace) {
	////ASSERT(is_node == false);
	////}

	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_DIV
	////result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = b

	return result
}

func Div(ctx *Context, a, b *Tensor) *Tensor {
	return DivImpl(ctx, a, b, false)
}

func DivInplace(ctx *Context, a, b *Tensor, inplace bool) *Tensor {
	return DivImpl(ctx, a, b, true)
}

// ggml_sgn
func SgnImpl(ctx *Context, a *Tensor, inplace bool) *Tensor {
	isNode := false

	if !inplace && a.grad != nil {
		isNode = true
	}

	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_SGN
	result.src0 = a
	result.src1 = nil

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

func Sgn(ctx *Context, a *Tensor) *Tensor {
	return SgnImpl(ctx, a, false)
}

func SgnInplace(ctx *Context, a *Tensor) *Tensor {
	return SgnImpl(ctx, a, true)
}

// struct ggml_tensor * Repeat(
func Repeat(ctx *Context, a, b *Tensor) *Tensor {
	////ASSERT(ggml_can_repeat(a, b));

	isNode := false

	if a.grad != nil {
		isNode = true
	}

	if AreSameShape(a, b) && !isNode {
		return a
	}

	result := NewTensor(ctx, a.Type, b.Dims, b.NE[0], b.NE[1], b.NE[2], b.NE[3], nil, nil) // Reusable OK

	result.op = OP_REPEAT
	result.src0 = a
	result.src1 = b

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

func IsScalar(tensor *Tensor) bool {
	return tensor.NE[0] == 1 && tensor.NE[1] == 1 && tensor.NE[2] == 1 && tensor.NE[3] == 1
}

func IsVector(tensor *Tensor) bool {
	return tensor.NE[1] == 1 && tensor.NE[2] == 1 && tensor.NE[3] == 1
}

func IsMatrix(tensor *Tensor) bool {
	return tensor.NE[2] == 1 && tensor.NE[3] == 1
}

// ggml_get_rows
func GetRows(ctx *Context, a, b *Tensor) *Tensor {
	////ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b.type == TYPE_I32);
	//if !IsMatrix(a) || !IsVector(b) /* || b.Type != TYPE_I32 */ {
	//	fmt.Printf("\n[ERROR] GetRows fail basic assertions")
	//	os.Exit(1)
	//}

	isNode := false

	if a.grad != nil || b.grad != nil {
		////ASSERT(false); // TODO: implement backward
		isNode = true
		fmt.Printf("\n[STOP] ml.GetRows")
		os.Exit(1)
	}

	result := NewTensor2D(ctx, TYPE_I8, a.NE[0], b.NE[0]) // Reusable OK

	result.op = OP_GET_ROWS
	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	result.src0 = a
	result.src1 = b

	return result
}

// ggml_get_rows
func GetRowsI8(ctx *Context, a, b *Tensor) *Tensor {
	////ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b.type == TYPE_I32);
	//if !IsMatrix(a) || !IsVector(b) /* || b.Type != TYPE_I32 */ {
	//	fmt.Printf("\n[ERROR] GetRows fail basic assertions")
	//	os.Exit(1)
	//}

	isNode := false

	if a.grad != nil || b.grad != nil {
		////ASSERT(false); // TODO: implement backward
		isNode = true
		fmt.Printf("\n[STOP] ml.GetRows")
		os.Exit(1)
	}

	result := NewTensor2D(ctx, TYPE_I8, a.NE[0], b.NE[0]) // Reusable OK

	result.op = OP_GET_ROWS
	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	result.src0 = a
	result.src1 = b

	return result
}

func RMSNorm(ctx *Context, a *Tensor) *Tensor {
	return RMSNormImpl(ctx, a, false)
}

func RMSNormInplace(ctx *Context, a *Tensor) *Tensor {
	return RMSNormImpl(ctx, a, true)
}

// ggml_rms_norm_impl
func RMSNormImpl(ctx *Context, a *Tensor, inplace bool) *Tensor {
	isNode := false

	if !inplace && a.grad != nil {
		////ASSERT(false); // TODO: implement backward
		isNode = true
		fmt.Printf("\n[STOP] ml.GetRows")
		os.Exit(1)
	}

	////struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_RMS_NORM
	result.src0 = a
	result.src1 = nil // TODO: maybe store epsilon here?

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

// ggml_view_1d
// NB! Originally offset in bytes, but here in floats (4-bytes)
func View1D(ctx *Context, a *Tensor, ne0 uint32, offset uint32) *Tensor {
	////if a.grad != nil {
	////	////ASSERT(false); // gradient propagation is not supported
	////	fmt.Printf("\n[STOP] View1D : gradient propagation is not supported")
	////	os.Exit(1)
	////}

	scalar := a.Scalars[offset/BLCK_SIZE[a.Type]:]
	slice := a.Data[offset:]
	result := NewTensor(ctx, a.Type, 1, ne0, 1, 1, 1, scalar, slice)

	result.op = OP_VIEW
	result.grad = nil
	result.src0 = a
	result.src1 = nil // TODO: maybe store the offset here?

	return result
}

// ggml_build_forward_impl
func BuildForwardImpl(graph *Graph, tensor *Tensor, expand bool) {

	if !expand {
		graph.NodesCount = 0
		graph.LeafsCount = 0
	}

	n0 := graph.NodesCount
	VisitParents(graph, tensor)
	n_new := graph.NodesCount - n0

	if n_new > 0 {
		// the last added node should always be starting point
		////ASSERT(cgraph.nodes[cgraph.n_nodes - 1] == tensor);
		if !(graph.Nodes[graph.NodesCount-1] == tensor) {
			fmt.Printf("\n[STOP] BuildForwardImpl : the last added node should always be starting point!")
			os.Exit(1)
		}
	}
}

// ggml_build_forward_expand
func BuildForwardExpand(graph *Graph, tensor *Tensor) {
	BuildForwardImpl(graph, tensor, true)
}

// ggml_visit_parents
func VisitParents(graph *Graph, node *Tensor) {

	if node.grad == nil {
		// this usually happens when we generate intermediate nodes from constants in the backward pass
		// it can also happen during forward pass, if the user performs computations with constants
		if node.op != OP_NONE {
			//PRINT_DEBUG("%s: warning: node %p has no grad, but op %d\n", __func__, (void *) node, node.op);
		}
	}

	// check if already visited
	for i := uint32(0); i < graph.NodesCount; i++ {
		if graph.Nodes[i] == node {
			return
		}
	}

	for i := uint32(0); i < graph.LeafsCount; i++ {
		if graph.Leafs[i] == node {
			return
		}
	}

	if node.src0 != nil {
		VisitParents(graph, node.src0)
	}

	if node.src1 != nil {
		VisitParents(graph, node.src1)
	}

	for i := 0; i < MAX_OPT; i++ {
		if node.opt[i] != nil {
			VisitParents(graph, node.opt[i])
		}
	}

	if node.op == OP_NONE && node.grad == nil {
		// reached a leaf node, not part of the gradient graph (e.g. a constant)
		////ASSERT(cgraph.n_leafs < MAX_NODES);

		graph.Leafs[graph.LeafsCount] = node
		graph.LeafsCount++
	} else {
		////ASSERT(cgraph.n_nodes < MAX_NODES);

		graph.Nodes[graph.NodesCount] = node
		graph.Grads[graph.NodesCount] = node.grad
		graph.NodesCount++
	}
}

// ggml_cpy
func CopyImpl(ctx *Context, a, b *Tensor, inplace bool) *Tensor {

	////ASSERT(ggml_nelements(a) == ggml_nelements(b));
	//if a.Nelements() != b.Nelements() {
	//	fmt.Printf("\n[HALT] Copy tensors of different dimensions!")
	//	os.Exit(1)
	//}

	isNode := false

	if !inplace && (a.grad != nil || b.grad != nil) {
		////ASSERT(false); // TODO: implement backward
		isNode = true
		fmt.Printf("\n[STOP] cpyImpl")
		os.Exit(1)
	}

	// make a view of the destination
	result := ViewTensor(ctx, b)

	result.op = OP_CPY
	result.src0 = a
	result.src1 = b

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

func Copy(ctx *Context, a, b *Tensor) *Tensor {
	return CopyImpl(ctx, a, b, false)
}

func CopyInplace(ctx *Context, a, b *Tensor) *Tensor {
	return CopyImpl(ctx, a, b, true)
}

// ggml_new_tensor_1d
func NewTensor1D(ctx *Context, dt DType, ne0 uint32) *Tensor {
	return NewTensor(ctx, dt, 1, ne0, 1, 1, 1, nil, nil)
}

// ggml_new_tensor_2d
func NewTensor2D(ctx *Context, dt DType, ne0, ne1 uint32) *Tensor {
	return NewTensor(ctx, dt, 2, ne0, ne1, 1, 1, nil, nil)
}

func NewTensor3D(ctx *Context, dt DType, ne0, ne1, ne2 uint32) *Tensor {
	return NewTensor(ctx, dt, 3, ne0, ne1, ne2, 1, nil, nil)
}

func NewTensor4D(ctx *Context, dt DType, ne0, ne1, ne2, ne3 uint32) *Tensor {
	return NewTensor(ctx, dt, 4, ne0, ne1, ne2, ne3, nil, nil)
}

func HydrateTensorFromFP32(t *Tensor, values []float32) {
	// implement
}

func HydrateTensorFromUI32(t *Tensor, values []uint32) {
	blockSize := BLCK_SIZE[t.Type]
	for i := 0; i < len(values); i += int(blockSize) {
		end := i + int(blockSize)
		if end > len(values) {
			end = len(values)
		}
		block := values[i:end]

		// Compute the max absolute value as scalar
		maxAbs := float16.Float16(0.0)
		for _, token := range block {
			val := float32(token)
			if absVal := float16.Float16(math.Abs(float64(val))); absVal > maxAbs {
				maxAbs = absVal
			}
		}
		if maxAbs == 0.0 {
			maxAbs = 1.0 // Avoid divide-by-zero
		}
		t.Scalars = append(t.Scalars, maxAbs)

		// Normalize and append to Data
		for _, token := range block {
			val := float16.Float16(token) / maxAbs
			t.Data = append(t.Data, int8(val))
		}
	}
}

// ggml_new_tensor_impl
func NewTensor(ctx *Context, dt DType, dims uint32, ne0, ne1, ne2, ne3 uint32, scalars []float16.Float16, data []int8) *Tensor {

	// TODO: Check allowed data types on graph creation
	//if dt != TYPE_F32 && dt != TYPE_I32 {
	//	fmt.Printf("\n[ERROR] NewTensorImpl got not supported type : %d", dt)
	//	os.Exit(1)
	//}

	////ggml_assert_aligned(result);

	dataLength := uint32(0)
	if data == nil {
		dataLength = ne0 * ne1 * ne2 * ne3
		data = make([]int8, dataLength, dataLength)
	}

	if scalars == nil {
		scalarLength := uint32(math.Ceil(float64(dataLength) / float64(BLCK_SIZE[dt])))
		scalars = make([]float16.Float16, scalarLength, scalarLength)
	}

	//if len(scalars) == 0 || (len(data)/len(scalars)) != 32 {
	//	fmt.Println(ne0)
	//	fmt.Println(ne1)
	//	fmt.Println(ne2)
	//	fmt.Println(ne3)
	//	fmt.Println("Invalid format . scalars for . values", len(scalars), len(data))
	//	pc, file, line, ok := runtime.Caller(2)
	//	if !ok {
	//		fmt.Println("Could not get caller info")
	//		os.Exit(0)
	//	}
	//
	//	fn := runtime.FuncForPC(pc)
	//	fmt.Printf("Called from %s:%d (%s)\n", file, line, fn.Name())
	//	os.Exit(0)
	//}

	return &Tensor{
		Type:    dt,
		Dims:    dims,
		NE:      [4]uint32{ne0, ne1, ne2, ne3},
		NB:      [4]uint32{TYPE_SIZE[dt], ne0 * TYPE_SIZE[dt], ne0 * ne1 * TYPE_SIZE[dt], ne0 * ne1 * ne2 * TYPE_SIZE[dt]},
		op:      OP_NONE,
		Scalars: scalars,
		Data:    data,
	}
}

// ggml_permute
func Permute(ctx *Context, a *Tensor, axis0, axis1, axis2, axis3 uint32) *Tensor {

	////ASSERT(axis0 >= 0 && axis0 < MAX_DIMS);
	////ASSERT(axis1 >= 0 && axis1 < MAX_DIMS);
	////ASSERT(axis2 >= 0 && axis2 < MAX_DIMS);
	////ASSERT(axis3 >= 0 && axis3 < MAX_DIMS);

	////ASSERT(axis0 != axis1);
	////ASSERT(axis0 != axis2);
	////ASSERT(axis0 != axis3);
	////ASSERT(axis1 != axis2);
	////ASSERT(axis1 != axis3);
	////ASSERT(axis2 != axis3);

	isNode := false

	if a.grad != nil {
		////ASSERT(false); // TODO: implement backward
		isNode = true
		fmt.Printf("\n[STOP] Permute error")
		os.Exit(1)
	}

	result := ViewTensor(ctx, a)

	var ne [MAX_DIMS]uint32
	var nb [MAX_DIMS]uint32

	ne[axis0] = a.NE[0]
	ne[axis1] = a.NE[1]
	ne[axis2] = a.NE[2]
	ne[axis3] = a.NE[3]

	nb[axis0] = a.NB[0]
	nb[axis1] = a.NB[1]
	nb[axis2] = a.NB[2]
	nb[axis3] = a.NB[3]

	result.NE[0] = ne[0]
	result.NE[1] = ne[1]
	result.NE[2] = ne[2]
	result.NE[3] = ne[3]

	result.NB[0] = nb[0]
	result.NB[1] = nb[1]
	result.NB[2] = nb[2]
	result.NB[3] = nb[3]

	result.op = OP_PERMUTE
	result.src0 = a
	result.src1 = nil // TODO: maybe store the permutation here?

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

// ggml_rope
func Rope(ctx *Context, a *Tensor, past, dims, mode uint32) *Tensor {
	////ASSERT(n_past >= 0);

	isNode := false

	if a.grad != nil {
		////ASSERT(false); // TODO: implement backward
		isNode = true
		fmt.Printf("\n[STOP] Rope error")
		os.Exit(1)
	}

	// TODO: when implement backward, fix this:
	//struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
	result := ViewTensor(ctx, a)

	b := NewTensor1D(ctx, TYPE_I32, 3)
	b.Data[0] = int8(past)
	b.Data[1] = int8(dims)
	b.Data[2] = int8(mode)

	result.op = OP_ROPE
	result.src0 = a
	result.src1 = b

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

func Reshape3D(ctx *Context, a *Tensor, ne0, ne1, ne2 uint32) *Tensor {
	////ASSERT(ggml_is_contiguous(a));
	////ASSERT(ggml_nelements(a) == ne0*ne1*ne2);

	//if !a.IsContiguous() {
	//	fmt.Printf("\n[STOP] Reshape3D : tensor is NOT contiguous!")
	//	os.Exit(1)
	//}

	//if a.Nelements() != ne0*ne1*ne2 {
	//	fmt.Printf("\n[STOP] Reshape3D : different elements number!")
	//	os.Exit(1)
	//}

	////bool is_node = false;

	////if (a.grad) {
	////   //// ASSERT(false); // TODO: implement backward
	////    is_node = true;
	////}

	result := NewTensor(ctx, a.Type, 3, ne0, ne1, ne2, 1, a.Scalars, a.Data) // Reusable OK

	result.op = OP_RESHAPE
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = nil

	return result
}

//// ggml_new_f32
//func NewFP32(ctx *Context, value float32) *Tensor {
//	result := NewTensor1D(ctx, TYPE_F32, 1) // Reusable OK
//	SetFP32(result, value)
//	return result
//}
//
//// ggml_set_f32
//func SetFP32(tensor *Tensor, value float32) *Tensor {
//	// FIXME Optimize with mem zeroing
//	n := tensor.Nelements()
//	for i := uint32(0); i < n; i++ {
//		////ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
//		tensor.Data[i] = value
//	}
//	return tensor
//}

// // ggml_new_I8
func NewI8(ctx *Context, scalar float32, value int8) *Tensor {
	result := NewTensor1D(ctx, TYPE_I8, 1) // Reusable OK
	SetI8(result, scalar, value)
	return result
}

// ggml_set_I8
func SetI8(tensor *Tensor, scalar float32, value int8) *Tensor {
	// FIXME Optimize with mem zeroing
	n := tensor.Nelements()
	for i := uint32(0); i < n; i++ {
		////ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
		//tensor.Scalars[i/32] = scalar
		tensor.Data[i] = value
	}
	return tensor
}

// ggml_scale
func ScaleImpl(ctx *Context, a, b *Tensor, inplace bool) *Tensor {
	////ASSERT(ggml_is_scalar(b));
	////ASSERT(ggml_is_padded_1d(a));

	////bool is_node = false;

	if !inplace && (a.grad != nil || b.grad != nil) {
		////ASSERT(false); // TODO: implement backward
		////is_node = true;
		fmt.Printf("\n[STOP] ScaleImpl : assertion failed")
		os.Exit(1)
	}

	// TODO: when implement backward, fix this:
	//struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
	result := ViewTensor(ctx, a)

	result.op = OP_SCALE
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = b

	return result
}

func Scale(ctx *Context, a, b *Tensor) *Tensor {
	return ScaleImpl(ctx, a, b, false)
}

func ScaleInplace(ctx *Context, a, b *Tensor) *Tensor {
	return ScaleImpl(ctx, a, b, true)
}

// ggml_diag_mask_inf
func DiagMaskInf(ctx *Context, a *Tensor, past uint32) *Tensor {
	////bool is_node = false;

	if a.grad != nil {
		////ASSERT(false); // TODO: implement backward
		////is_node = true;
		fmt.Printf("\n[STOP] DiagMaskInf : assertion failed")
		os.Exit(1)
	}

	// TODO: when implement backward, fix this:
	//struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
	result := ViewTensor(ctx, a)
	b := NewI8(ctx, 1.0, int8(past)) // FIXME NewI32(ctx, past)

	result.op = OP_DIAG_MASK_INF
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = b

	return result
}

// ggml_soft_max
func SoftMax(ctx *Context, a *Tensor) *Tensor {
	////bool is_node = false;

	if a.grad != nil {
		////ASSERT(false); // TODO: implement backward
		////is_node = true;
		fmt.Printf("\n[STOP] SoftMax : assertion failed")
		os.Exit(1)
	}

	// TODO: when implement backward, fix this:
	//struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
	result := ViewTensor(ctx, a)

	result.op = OP_SOFT_MAX
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = nil

	return result
}

// ggml_silu

func SiluImpl(ctx *Context, a *Tensor, inplace bool) *Tensor {
	////bool is_node = false;

	////if (!inplace && (a.grad)) {
	////is_node = true;
	////}

	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_SILU
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = nil

	return result
}

func Silu(ctx *Context, a *Tensor) *Tensor {
	return SiluImpl(ctx, a, false)
}

func SiluInplace(ctx *Context, a *Tensor) *Tensor {
	return SiluImpl(ctx, a, true)
}

// ggml_step
func StepImpl(ctx *Context, a *Tensor, inplace bool) *Tensor {
	isNode := false

	if !inplace && a.grad != nil {
		isNode = true
	}

	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_STEP
	result.src0 = a
	result.src1 = nil

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

func Step(ctx *Context, a *Tensor) *Tensor {
	return StepImpl(ctx, a, false)
}

func StepInplace(ctx *Context, a *Tensor) *Tensor {
	return StepImpl(ctx, a, true)
}

// ggml_transpose

func Transpose(ctx *Context, a *Tensor) *Tensor {
	////isNode := false

	if a.grad != nil {
		////ASSERT(false); // TODO: implement backward
		////is_node = true;
	}

	result := ViewTensor(ctx, a)

	result.NE[0] = a.NE[1]
	result.NE[1] = a.NE[0]

	result.NB[0] = a.NB[1]
	result.NB[1] = a.NB[0]

	result.op = OP_TRANSPOSE
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = nil

	return result
}

func BuildForward(tensor *Tensor) *Graph {
	result := Graph{}
	BuildForwardImpl(&result, tensor, false)
	return &result
}

func BuildBackward(ctx *Context, gf *Graph, keep bool) Graph {

	result := *gf
	////ASSERT(gf.n_nodes > 0);

	// if we are keeping the gradient graph, we have to detach the gradient nodes from the original graph
	if keep {
		for i := uint32(0); i < gf.NodesCount; i++ {
			node := gf.Nodes[i]

			if node.grad != nil {
				node.grad = DupTensor(ctx, node)
				gf.Grads[i] = node.grad
			}
		}
	}

	for i := gf.NodesCount - 1; i >= 0; i-- {
		node := gf.Nodes[i]

		// because we detached the grad nodes from the original graph, we can afford inplace operations
		if node.grad != nil {
			ComputeBackward(ctx, node, keep)
		}
	}

	for i := gf.NodesCount - 1; i >= 0; i-- {
		node := gf.Nodes[i]

		if node.isParam {
			////PRINT_DEBUG("%s: found root node %p\n", __func__, (void *) node);
			BuildForwardImpl(&result, node.grad, true)
		}
	}

	return result
}

////////////////////////////////////////////////////////////////////////////////

func ComputeBackward(ctx *Context, tensor *Tensor, inplace bool) {

	src0 := tensor.src0
	src1 := tensor.src1

	switch tensor.op {

	case OP_DUP:
		if src0.grad != nil {
			src0.grad = AddImpl(ctx, src0.grad, tensor.grad, inplace)
		}
	case OP_ADD:
		if src0.grad != nil {
			src0.grad = AddImpl(ctx, src0.grad, tensor.grad, inplace)
		}
		if src1.grad != nil {
			src1.grad = AddImpl(ctx, src1.grad, tensor.grad, inplace)
		}
	case OP_SUB:
		if src0.grad != nil {
			src0.grad = AddImpl(ctx, src0.grad, tensor.grad, inplace)
		}
		if src1.grad != nil {
			src1.grad = SubImpl(ctx, src1.grad, tensor.grad, inplace)
		}
	case OP_MUL:
		if src0.grad != nil {
			src0.grad =
				AddImpl(ctx,
					src0.grad,
					Mul(ctx, src1, tensor.grad),
					inplace)
		}
		if src1.grad != nil {
			src1.grad =
				AddImpl(ctx,
					src1.grad,
					Mul(ctx, src0, tensor.grad),
					inplace)
		}
	case OP_DIV:
		if src0.grad != nil {
			src0.grad =
				AddImpl(ctx,
					src0.grad,
					Div(ctx, tensor.grad, src1),
					inplace)
		}
		if src1.grad != nil {
			src1.grad =
				SubImpl(ctx,
					src1.grad,
					Mul(ctx,
						tensor.grad,
						Div(ctx, tensor, src1)),
					inplace)
		}
	case OP_SQR:
		if src0.grad != nil {
			src0.grad =
				AddImpl(ctx,
					src0.grad,
					Mul(ctx,
						Mul(ctx, src0, tensor.grad),
						Repeat(ctx, NewI8(ctx, 2.0, 1), src0)),
					inplace)
		}
	case OP_SQRT:
		if src0.grad != nil {
			src0.grad =
				AddImpl(ctx,
					src0.grad,
					Div(ctx,
						Repeat(ctx, NewI8(ctx, 0.5, 1), tensor),
						tensor),
					inplace)
		}
	case OP_SUM:
		if src0.grad != nil {
			src0.grad =
				AddImpl(ctx,
					src0.grad,
					Repeat(ctx, tensor.grad, src0.grad),
					inplace)
		}
	case OP_MEAN:
		//// ASSERT(false); // TODO: implement
	case OP_REPEAT:
		if src0.grad != nil {
			src0.grad =
				AddImpl(ctx,
					src0.grad,
					Sum(ctx, tensor.grad),
					inplace)
		}
	case OP_ABS:
		if src0.grad != nil {
			src0.grad =
				AddImpl(ctx,
					src0.grad,
					Mul(ctx,
						Sgn(ctx, src0),
						tensor.grad),
					inplace)
		}
	case OP_SGN:
		if src0.grad != nil {
			// noop
		}
	case OP_NEG:
		if src0.grad != nil {
			src0.grad = SubImpl(ctx, src0.grad, tensor.grad, inplace)
		}
	case OP_STEP:
		if src0.grad != nil {
			// noop
		}
	case OP_RELU:
		if src0.grad != nil {
			src0.grad = SubImpl(ctx,
				src0.grad,
				Mul(ctx,
					Step(ctx, src0),
					tensor.grad),
				inplace)
		}
	case OP_GELU:
		//// ASSERT(false); // TODO: not implemented
	case OP_SILU:
		//// ASSERT(false); // TODO: not implemented
	case OP_NORM:
		//// ASSERT(false); // TODO: not implemented
	case OP_RMS_NORM:
		//// ASSERT(false); // TODO: not implemented
	case OP_MUL_MAT:
		if src0.grad != nil {
			// TODO: this requires outer product - ggml_out_prod(ctx, src1, tensor.grad);
			//// ASSERT(false);
			fmt.Printf("\n[HALT] ComputeBackward : OP_MUL_MAT with src0.grad!")
			os.Exit(1)
		}
		if src1.grad != nil {
			src1.grad =
				AddImpl(ctx,
					src1.grad,
					// TODO: fix transpose, the node will break the graph connections
					MulMat(ctx, Transpose(ctx, src0), tensor.grad),
					inplace)
		}
	case OP_SCALE:
		//// ASSERT(false); // TODO: not implemented
	case OP_CPY:
		//// ASSERT(false); // TODO: not implemented
	case OP_RESHAPE:
		//// ASSERT(false); // TODO: not implemented
	case OP_VIEW:
		//// ASSERT(false); // not supported
	case OP_PERMUTE:
		//// ASSERT(false); // TODO: not implemented
	case OP_TRANSPOSE:
		//// ASSERT(false); // TODO: not implemented
	case OP_GET_ROWS:
		//// ASSERT(false); // TODO: not implemented
	case OP_DIAG_MASK_INF:
		//// ASSERT(false); // TODO: not implemented
	case OP_SOFT_MAX:
		//// ASSERT(false); // TODO: not implemented
	case OP_ROPE:
		//// ASSERT(false); // TODO: not implemented
	case OP_CONV_1D_1S:
		//// ASSERT(false); // TODO: not implemented
	case OP_CONV_1D_2S:
		//// ASSERT(false); // TODO: not implemented
	case OP_FLASH_ATTN:
		//// ASSERT(false); // not supported
	case OP_FLASH_FF:
		//// ASSERT(false); // not supported
	case OP_NONE:
		// nop
	case OP_COUNT:
		//// ASSERT(false);
	}
}

// ---

type TaskType uint8

const (
	TASK_INIT     TaskType = 0
	TASK_COMPUTE  TaskType = 1
	TASK_FINALIZE TaskType = 2
)

type ComputeParams struct {
	Type TaskType

	ith uint32
	nth uint32

	tensor *Tensor

	wg *sync.WaitGroup

	UseAVX  bool
	UseNEON bool
}

// Golang doesn’t have unary Bitwise NOT(~) like other programming languages
// Here, you have to use Bitwise XOR(^) operator as Bitwise NOT
func up32(n uint32) uint32 { // FIXME Not needed ?
	return uint32(n+31) & ^uint32(31)
}

func up(n, m uint32) uint32 { // FIXME Not needed ?
	// assert m is a power of 2
	////GGML_ASSERT((m & (m - 1)) == 0);
	return uint32(n+m-1) & ^uint32(m-1)
}

func max(a, b int) int { // FIXME Not needed ?
	if a >= b {
		return a
	}
	return b
}

// Job is goroutine existing while the computation loop is active
// The main purpose of the Job is to perform some part
// of time consuming matrix multiplications
// TODO: Investigate https://pkg.go.dev/runtime#LockOSThread
func Job(listen <-chan *ComputeParams, id int) {
	runtime.LockOSThread()
	for params := range listen {
		ComputeForwardMulMatI8(
			params,
			params.tensor.src0,
			params.tensor.src1,
			params.tensor)
		params.wg.Done()
	}
}

// Do is an experimental alternative for always waiting Job threads
func Do(params *ComputeParams, id int) {
	ComputeForwardMulMatI8(
		params,
		params.tensor.src0,
		params.tensor.src1,
		params.tensor)
	params.wg.Done()
}

func GraphCompute(ctx *Context, graph *Graph) {

	//maxThreads := graph.MaxThreads
	maxThreads := ctx.MaxThreads

	// --- init N job goroutines and channel to send tasks for them

	//graph.Jobs = make(chan *ComputeParams, maxThreads) // TODO Right place to init? +1 for safety?
	//defer close(graph.Jobs)

	//for i := 0; i < maxThreads; i++ {
	//	go Job(graph.Jobs, i)
	//}

	// --- initialize tasks

	{
		// thread scheduling for the different operations
		// TasksCount might be 0, 1, or ThreadsCount
		for i := uint32(0); i < graph.NodesCount; i++ {

			node := graph.Nodes[i]

			if DEBUG {
				fmt.Printf("\n\n### STEP #%d ### %d - %d [ %d:%d:%d:%d ]", i, node.op, node.Type, node.NE[0], node.NE[1], node.NE[2], node.NE[3])
			}

			switch node.op {

			case OP_DUP:
				node.TasksCount = 1
			case OP_ADD:
				node.TasksCount = 1 // TODO threads
			case OP_SUB:
			case OP_MUL:
			case OP_DIV:
			case OP_SQR:
			case OP_SQRT:
			case OP_SUM:
			case OP_MEAN:
			case OP_REPEAT:
			case OP_ABS:
			case OP_SGN:
			case OP_NEG:
			case OP_STEP:
			case OP_RELU:
				node.TasksCount = 1
			case OP_GELU:
				node.TasksCount = 1 // TODO threads
			case OP_SILU:
				node.TasksCount = 1 // TODO threads
			case OP_NORM:
			case OP_RMS_NORM:
				node.TasksCount = 1 // TODO threads
			case OP_MUL_MAT:
				node.TasksCount = maxThreads
				// TODO: use different scheduling for different matrix sizes
			case OP_SCALE:
				node.TasksCount = 1 // TODO threads
			case OP_CPY:
			case OP_RESHAPE:
			case OP_VIEW:
			case OP_PERMUTE:
			case OP_TRANSPOSE:
			case OP_GET_ROWS:
			case OP_DIAG_MASK_INF:
				node.TasksCount = 1
			case OP_SOFT_MAX:
				node.TasksCount = 1 // TODO threads
			case OP_ROPE:
				////node.TasksCount = 1
			case OP_CONV_1D_1S:
			case OP_CONV_1D_2S:
				node.TasksCount = 1 // TODO threads
				////ASSERT(node->src0->ne[3] == 1);
				////ASSERT(node->src1->ne[2] == 1);
				////ASSERT(node->src1->ne[3] == 1);
			case OP_FLASH_ATTN:
				node.TasksCount = 1 // TODO threads
			case OP_FLASH_FF:
				node.TasksCount = 1 // TODO threads
			case OP_NONE:
				node.TasksCount = 1
			case OP_COUNT:
				fmt.Printf("\n[HALT] Something wrong with compute graph!")
				os.Exit(1)
			}
		}
	}

	//fmt.Println(graph.NodesCount)
	// 1253

	for i := uint32(0); i < graph.NodesCount; i++ {

		node := graph.Nodes[i]

		if DEBUG {
			fmt.Printf("\n\n### STEP #%d ### %d - %d [ %d:%d:%d:%d ]", i, node.op, node.Type, node.NE[0], node.NE[1], node.NE[2], node.NE[3])
		}

		params := &ComputeParams{
			Type: TASK_INIT,
			ith:  0,
			nth:  uint32(node.TasksCount),
		}

		ComputeForward(ctx, graph, params, node) // TASK_INIT

		// --- COMPUTE

		params.Type = TASK_COMPUTE
		ComputeForward(ctx, graph, params, node)

		if (graph.NodesCount % 1) == 0 {
			fmt.Println("Did forward operation... Now at: Node", i, "with", i/graph.NodesCount*100, "%", node.op)
		}
		// --- FINALIZE

		params.Type = TASK_FINALIZE
		ComputeForward(ctx, graph, params, node)
	}

}

// =======================================================================

func ComputeForward(ctx *Context, graph *Graph, params *ComputeParams, tensor *Tensor) {

	switch tensor.op {

	case OP_DUP:
		////ggml_compute_forward_dup(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_dup")
		os.Exit(1)
	case OP_ADD:
		ComputeForwardAddI8(params, tensor.src0, tensor.src1, tensor)
	case OP_SUB:
		////ggml_compute_forward_sub(params, tensor->src0, tensor->src1, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_sub")
		os.Exit(1)
	case OP_MUL:
		ComputeForwardMulI8(params, tensor.src0, tensor.src1, tensor)
	case OP_DIV:
		////ggml_compute_forward_div(params, tensor->src0, tensor->src1, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_div")
		os.Exit(1)
	case OP_SQR:
		////ggml_compute_forward_sqr(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_sqr")
		os.Exit(1)
	case OP_SQRT:
		////ggml_compute_forward_sqrt(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_sqrt")
		os.Exit(1)
	case OP_SUM:
		////ggml_compute_forward_sum(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_sum")
		os.Exit(1)
	case OP_MEAN:
		////ggml_compute_forward_mean(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_mean")
		os.Exit(1)
	case OP_REPEAT:
		ComputeForwardRepeatI8(params, tensor.src0, tensor)
	case OP_ABS:
		////ggml_compute_forward_abs(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_abs")
		os.Exit(1)
	case OP_SGN:
		////ggml_compute_forward_sgn(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_sgn")
		os.Exit(1)
	case OP_NEG:
		////ggml_compute_forward_neg(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_neg")
		os.Exit(1)
	case OP_STEP:
		////ggml_compute_forward_step(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_step")
		os.Exit(1)
	case OP_RELU:
		////ggml_compute_forward_relu(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_relu")
		os.Exit(1)
	case OP_GELU:
		////ggml_compute_forward_gelu(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_gelu")
		os.Exit(1)
	case OP_SILU:
		ComputeForwardSiluI8(params, tensor.src0, tensor)
	case OP_NORM:
		////ggml_compute_forward_norm(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_norm")
		os.Exit(1)
	case OP_RMS_NORM:
		ComputeForwardRMSNormI8(params, tensor.src0, tensor)
	case OP_MUL_MAT:

		// TODO Optimize this
		if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
			return
		}

		// FIXME: Need better heuristic for how many threads to use there
		// But not more than minimal dimension of tensors involved!
		// Like if there dim = 8, it safe to use only 8 or less threads, not 12

		// TODO: There might be small architectures where not reasonable to spin up
		// all available threads, so better to limit parallelism here
		// But that's not the case for LLMs and particularly LLaMA, thus commented

		// totalRows := tensor.src0.NE[1] * tensor.src0.NE[2] * tensor.src0.NE[3]
		// maxThreads := min(graph.MaxThreads, int(totalRows))

		//maxThreads := graph.MaxThreads
		maxThreads := ctx.MaxThreads

		wg := new(sync.WaitGroup)
		wg.Add(maxThreads)

		for i := 0; i < maxThreads; i++ {

			//graph.Jobs <- &ComputeParams{
			ctx.Compute <- &ComputeParams{
				Type:   TASK_COMPUTE,
				ith:    uint32(i),
				nth:    uint32(maxThreads),
				tensor: tensor,
				//UseNEON: graph.UseNEON,
				UseNEON: ctx.UseNEON,
				//UseAVX:  graph.UseAVX,
				UseAVX: ctx.UseAVX,
				wg:     wg,
			}

			//go Do(&ComputeParams{
			//	Type:    TASK_COMPUTE,
			//	ith:     uint32(i),
			//	nth:     uint32(maxThreads),
			//	tensor:  tensor,
			//	UseNEON: graph.UseNEON,
			//	UseAVX:  graph.UseAVX,
			//	wg:      wg,
			//}, i)
		}

		wg.Wait()

	case OP_SCALE:
		ComputeForwardScaleI8(params, tensor.src0, tensor.src1, tensor)
	case OP_CPY:
		ComputeForwardDupI8(params, tensor.src0, tensor)
	case OP_RESHAPE:
		ComputeForwardReshape(params, tensor.src0, tensor) // NOP
	case OP_VIEW:
		ComputeForwardView(params, tensor.src0) // NOP
	case OP_PERMUTE:
		ComputeForwardPermute(params, tensor.src0) // NOP
	case OP_TRANSPOSE:
		////ggml_compute_forward_transpose(params, tensor->src0);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_transpose")
		os.Exit(1)
	case OP_GET_ROWS:
		ComputeForwardGetRows(params, tensor.src0, tensor.src1, tensor)
	case OP_DIAG_MASK_INF:
		ComputeForwardDiagMaskInfI8(params, tensor.src0, tensor.src1, tensor)
	case OP_SOFT_MAX:
		ComputeForwardSoftMaxI8(params, tensor.src0, tensor)
	case OP_ROPE:
		ComputeForwardRopeI8(params, tensor.src0, tensor.src1, tensor)
	case OP_CONV_1D_1S:
		////ggml_compute_forward_conv_1d_1s(params, tensor->src0, tensor->src1, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_conv_1d_1s")
		os.Exit(1)
	case OP_CONV_1D_2S:
		////ggml_compute_forward_conv_1d_2s(params, tensor->src0, tensor->src1, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_conv_1d_2s")
		os.Exit(1)
	case OP_FLASH_ATTN:
		////int32_t t = ggml_get_i32_1d(tensor->opt[1], 0);
		////ASSERT(t == 0 || t == 1);
		////bool masked = t != 0;
		////ggml_compute_forward_flash_attn(params, tensor->src0, tensor->src1, tensor->opt[0], masked, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_flash_attn")
		os.Exit(1)
	case OP_FLASH_FF:
		////ggml_compute_forward_flash_ff(params, tensor->src0, tensor->src1, tensor->opt[0], tensor->opt[1], tensor->opt[2], tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_flash_ff")
		os.Exit(1)
	case OP_NONE:
		// nop
	case OP_COUNT:
		////ASSERT(false);
		fmt.Printf("\n[HALT] ComputeForward got OP_COUNT method!")
		os.Exit(1)
	}
}

func VecCopyFP32(n uint32, y, x []float32) {
	for i := uint32(0); i < n; i++ {
		y[i] = x[i]
	}
}

func VecCopyI8(scalary float16.Float16, y []int8, scalarY float16.Float16, x []int8) {
	scalarY = scalarY

	for i := 0; i < len(x); i++ {
		y[i] = x[i]
	}
}

// ggml_compute_forward_get_rows_f32
func ComputeForwardGetRows(params *ComputeParams, src0, src1, dst *Tensor) {

	////assert(params->ith == 0);

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	nc := src0.NE[0]
	nr := src1.Nelements()

	if dst.NE[0] != nc || dst.NE[1] != nr || src0.NB[0] != TYPE_SIZE[TYPE_I8] {
		fmt.Printf("[HALT]ComputeForwardGetRows : wrong dimensions!")
		os.Exit(1)
	}

	numberOfChunks := uint32(math.Ceil(float64(nr) / 32))

	for c := uint32(0); c < numberOfChunks; c++ {

		VecCopyI8(
			dst.Scalars[c],
			dst.Data[c*32:(c+1)*32],
			src0.Scalars[c],
			src0.Data[c*32:(c+1)*32],
		) // TODO copy()
	}
}

// ggml_compute_forward_rms_norm_f32
func ComputeForwardRMSNormI8(params *ComputeParams, src0, dst *Tensor) {
	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	// Configure the destination tensor's metadata.
	dst.NE = src0.NE
	dst.NB = src0.NB
	dst.Type = TYPE_I8
	dst.op = OP_RMS_NORM

	// Get the dimensions and row count.
	nc := src0.NE[0]
	nr := src0.Nrows()

	// Distribute the rows across the available threads.
	nth := params.nth
	ith := params.ith
	rowsPerThread := (nr + nth - 1) / nth
	startRow := ith * rowsPerThread
	endRow := (ith + 1) * rowsPerThread
	if endRow > nr {
		endRow = nr
	}

	// Temporary buffer for a dequantized row, reused for each row processed by the thread.
	f32Row := make([]float16.Float16, nc)

	// Process the assigned rows.
	for r := startRow; r < endRow; r++ {
		rowStartIdx := r * nc

		// Step 1: Dequantize the current row from src0 into the temporary buffer.
		for c := uint32(0); c < nc; c++ {
			idx := rowStartIdx + c
			blockIdx := idx / 32
			scalar := src0.Scalars[blockIdx]
			f32Row[c] = float16.Float16(src0.Data[idx]) * scalar
		}

		// Step 2: Calculate the Root Mean Square (RMS) for the dequantized row.
		var ss float64
		for _, val := range f32Row {
			ss += float64(val * val)
		}
		ss /= float64(nc)
		invRMS := 1.0 / math.Sqrt(ss+1e-5)

		// Step 3: Normalize the row and scale it by the weights from src1.
		for c := uint32(0); c < nc; c++ {
			f32Row[c] = float16.Float16(invRMS) * f32Row[c]
		}

		// Step 4: Re-quantize the processed float32 row back into the int8 dst tensor.
		// This is done in blocks of 32 to compute a new scalar for each block.
		for c := uint32(0); c < nc; c += 32 {
			block := f32Row[c : c+32]
			var amax float16.Float16

			// Find the absolute maximum value in the block.
			for _, val := range block {
				if abs := float16.Float16(math.Abs(float64(val))); abs > amax {
					amax = abs
				}
			}

			// Calculate and store the new scalar for the destination block.
			newScalar := float16.Float16(0.0)
			if amax > 0 {
				newScalar = amax / 127.0
			}
			dstBlockIdx := (rowStartIdx + c) / 32
			dst.Scalars[dstBlockIdx] = newScalar

			// Quantize the float32 block and store it in the destination tensor.
			if newScalar == 0.0 {
				for i := uint32(0); i < 32; i++ {
					dst.Data[rowStartIdx+c+i] = 0
				}
			} else {
				invNewScalar := 1.0 / newScalar
				for i, val := range block {
					quantizedVal := math.Round(float64(val * invNewScalar))
					// Clamp the value to the int8 range.
					if quantizedVal > 127.0 {
						quantizedVal = 127.0
					} else if quantizedVal < -128.0 {
						quantizedVal = -128.0
					}
					dst.Data[rowStartIdx+c+uint32(i)] = int8(quantizedVal)
				}
			}
		}
	}
}

// ggml_vec_scale_f32
func VecScaleFP32(n uint32, y []float32, v float32) {
	for i := uint32(0); i < n; i++ {
		y[i] *= v
	}
}

// ggml_vec_scale_f32
func VecScaleI8(scalarY float16.Float16, v float16.Float16) {
	scalarY *= v
}

// ggml_compute_forward_repeat
func ComputeForwardRepeatI8(params *ComputeParams, src0, dst *Tensor) {

	// --- Repeat Quantized Data (int8) ---
	// This part remains the same as it would for float32 scalars.
	nSrcElements := src0.Nelements()
	nDstElements := dst.Nelements()
	repeats := nDstElements / nSrcElements

	srcData := src0.Data[:nSrcElements]
	dstData := dst.Data[:nDstElements]

	for i := uint32(0); i < repeats; i++ {
		copy(dstData[i*nSrcElements:], srcData)
	}

	// --- Repeat Quantization Scalars (float16) ---
	// The original function would work with float32 scalars. This version is
	// adapted for float16 scalars. For performance, we avoid per-element conversion
	// and instead reinterpret the memory of the slice.
	blockSize := BLCK_SIZE[TYPE_I8]
	nSrcBlocks := nSrcElements / blockSize
	nDstBlocks := nDstElements / blockSize

	if nSrcBlocks == 0 {
		return // Nothing to repeat
	}
	scalarRepeats := nDstBlocks / nSrcBlocks

	// To handle float16 scalars efficiently without changing the Tensor struct,
	// we reinterpret the []float32 slices as []float16.Float16 slices.
	// A float32 is 4 bytes, while a float16 is 2 bytes. We assume that the
	// `Scalars` slice was allocated with enough capacity to hold the required
	// number of float16 values.
	var srcScalarsF16 []float16.Float16
	srcHeader := (*reflect.SliceHeader)(unsafe.Pointer(&src0.Scalars))
	srcF16Header := (*reflect.SliceHeader)(unsafe.Pointer(&srcScalarsF16))
	srcF16Header.Data = srcHeader.Data
	srcF16Header.Len = int(nSrcBlocks)
	srcF16Header.Cap = int(nSrcBlocks)

	var dstScalarsF16 []float16.Float16
	dstHeader := (*reflect.SliceHeader)(unsafe.Pointer(&dst.Scalars))
	dstF16Header := (*reflect.SliceHeader)(unsafe.Pointer(&dstScalarsF16))
	dstF16Header.Data = dstHeader.Data
	dstF16Header.Len = int(nDstBlocks)
	dstF16Header.Cap = int(nDstBlocks)

	// Now copy the float16 scalars in bulk
	for i := uint32(0); i < scalarRepeats; i++ {
		copy(dstScalarsF16[i*nSrcBlocks:], srcScalarsF16)
	}
}

func VecMulFP32(n uint32, z, x, y []float32) {
	for i := uint32(0); i < n; i++ {
		z[i] = x[i] * y[i]
	}
}

func VecMulI8(scalarZ float16.Float16, z []int8, scalarX float16.Float16, x []int8, scalarY float16.Float16, y []int8) {
	if len(x) != len(y) {
		fmt.Printf("[HALT]VecMulI8 : x and y must have the same length!")
		os.Exit(1)
	}

	length := len(x)

	dstF32 := make([]float16.Float16, length)

	for i := uint32(0); i < uint32(length); i++ {
		dstF32[i] = (scalarX * float16.Float16(x[i])) * (scalarY * float16.Float16(y[i]))
	}

	maxValue := dstF32[0]
	for _, val := range dstF32 {
		if val > maxValue {
			maxValue = val
		}
	}

	scalarZ = float16.Float16(float32(maxValue) / 127)

	for i, val := range dstF32 {
		z[i] = int8(float32(val) / float32(scalarZ))
	}
}

// ggml_compute_forward_mul
func ComputeForwardMulI8(params *ComputeParams, src0, src1, dst *Tensor) {

	////assert(params->ith == 0);
	////assert(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

	if !AreSameShape(src0, src1) || !AreSameShape(src0, dst) {
		fmt.Printf("\n[HALT] ComputeForwardMulFP32 : different shapes!")
		os.Exit(1)
	}

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	n := src0.Nrows()
	//nc := src0.NE[0]

	////assert( dst->nb[0] == sizeof(float));
	////assert(src0->nb[0] == sizeof(float));
	////assert(src1->nb[0] == sizeof(float));

	for c := uint32(0); c < ((n / 32) + 1); c++ {
		maxNValues := uint32(min(32, int(n-(c*32))))

		VecMulI8(
			dst.Scalars[c],
			dst.Data[c*32:c*32+maxNValues],
			src0.Scalars[c],
			src0.Data[c*32:c*32+maxNValues],
			src1.Scalars[c],
			src1.Data[c*32:c*32+maxNValues],
		)

		//for i := uint32(0); i < maxNValues; i++ {
		//
		//	////ggml_vec_mul_f32(nc,
		//	////(float *) ((char *) dst->data  + i*( dst->nb[1])),
		//	////(float *) ((char *) src0->data + i*(src0->nb[1])),
		//	////(float *) ((char *) src1->data + i*(src1->nb[1])));
		//
		//	// FIXME NE vs NB
		//	//VecMulFP32(nc, dst.Data[i*dst.NE[0]:], src0.Data[i*src0.NE[0]:], src1.Data[i*src1.NE[0]:])
		//}
	}

	if DEBUG {
		printTensor(src0, "MUL SRC0")
		printTensor(src1, "MUL SRC1")
		printTensor(dst, "MUL DST")
	}
}

// ggml_vec_dot_f32
func VecDotFP32(n uint32, x, y []float32) float32 {
	sumf := float32(0.0)
	for i := uint32(0); i < n; i++ {
		sumf += x[i] * y[i]
	}
	return sumf
}

// ggml_vec_mad_f32
func VecMadFP32(n uint32, y, x []float32, v float32) {
	for i := uint32(0); i < n; i++ {
		y[i] += x[i] * v
	}
}

// ggml_vec_acc_f32
func VecAccFP32(n uint32, y, x []float32) {
	for i := uint32(0); i < n; i++ {
		y[i] += x[i]
	}
}

// TODO: Implement all the tensor asserts BEFORE the real computing
func CheckGraph() {

	// --- ComputeForwardMulMatFP32(params *ComputeParams, src0, src1, dst *Tensor)

	////assert(ne02 == ne12);
	////assert(ne03 == ne13);
	////assert(ne2  == ne12);
	////assert(ne3  == ne13);

	// TODO: we don't support permuted src0
	////assert(nb00 == sizeof(float) || nb01 == sizeof(float));

	// dst cannot be transposed or permuted
	////assert(nb0 == sizeof(float));
	////assert(nb0 <= nb1);
	////assert(nb1 <= nb2);
	////assert(nb2 <= nb3);

	////assert(ne0 == ne01);
	////assert(ne1 == ne11);
	////assert(ne2 == ne02);
	////assert(ne3 == ne03);

	// nb01 >= nb00 - src0 is not transposed
	//   compute by src0 rows

	// TODO: do not support transposed src1
	////assert(nb10 == sizeof(float));
	////if nb10 == 4 {
	////	fmt.Printf("\n[HALT] Do not support transposed src1")
	////	os.Exit(1)
	////}

}

// ggml_compute_forward_mul_mat_f32
func ComputeForwardMulMatI8(params *ComputeParams, src0, src1, dst *Tensor) {

	// --- Copy tensor parameters to local vars for compact fitting in CPU cache lines
	ne00 := src0.NE[0]
	ne01 := src0.NE[1]
	ne02 := src0.NE[2]
	ne03 := src0.NE[3]

	ne11 := src1.NE[1]

	nb0 := dst.NB[0]
	nb1 := dst.NB[1]
	nb2 := dst.NB[2]
	nb3 := dst.NB[3]
	nb01 := src0.NB[1]
	nb02 := src0.NB[2]
	nb03 := src0.NB[3]
	nb11 := src1.NB[1]
	nb12 := src1.NB[2]
	nb13 := src1.NB[3]

	// --- Threading logic from original function
	nr := ne01 * ne02 * ne03                 // total rows in src0
	dr := (nr + params.nth - 1) / params.nth // rows per thread
	ir0 := dr * params.ith                   // row range...
	ir1 := min32(ir0+dr, nr)                 // ...for this thread

	// For Q8_0, the block size is 32. Every 32 int8 values have one float16 scalar.
	const blockSize = 32

	mult := ne02 * ne01
	for ir := ir0; ir < ir1; ir++ {

		// original GGML indices math + bit optimizations
		//i03 := ir / (ne02 * ne01)
		i03 := ir / mult
		//i02 := (ir - i03*ne02*ne01) / ne01
		diff := ir - i03*mult
		//i02 := (ir - i03*mult) / ne01
		i02 := diff / ne01
		//i01 := (ir - i03*ne02*ne01 - i02*ne01)
		//i01 := ir - i03*mult - i02*ne01
		i01 := diff - i02*ne01

		src0Offset := i01*nb01 + i02*nb02 + i03*nb03

		for ic := uint32(0); ic < ne11; ic++ {

			src1Offet := ic*nb11 + i02*nb12 + i03*nb13
			dstOffset := i01*nb0 + ic*nb1 + i02*nb2 + i03*nb3

			// Dequantization scalars for the current block
			src0ScalarPtr := src0.Scalars[src0Offset/4/32]
			src1ScalarPtr := src1.Scalars[src1Offet/4/32]

			// Slices for the int8 data for the current block
			src0Ptr := src0.Data[src0Offset/4:]
			src1Ptr := src1.Data[src1Offet/4:]

			sum := float32(0.0)
			for i := uint32(0); i < ne00; i++ {
				sum += (float32(src0ScalarPtr) * float32(src0Ptr[i])) * (float32(src1ScalarPtr) * float32(src1Ptr[i]))
			}

			dst.Data[dstOffset/4] = int8(sum)
		}
	}
}

// min32 is a helper function to find the minimum of two uint32 values.
func min32(a, b uint32) uint32 {
	if a < b {
		return a
	}
	return b
}

// ggml_compute_forward_view
func ComputeForwardView(params *ComputeParams, src0 *Tensor) {
	// NOP
}

func ComputeForwardCopy(params *ComputeParams, src0, dst *Tensor) {
	ComputeForwardDupI8(params, src0, dst)
}

// ComputeForwardDupQ8 copies a tensor that is stored in Q8_0 layout:
//
//   - src0.Data    []int8            – 32 values per block
//   - src0.Scalars []float16.Float16 – 1 scale per block
//
// The destination tensor must have the same Q8_0 type.
//
// Thread-parallelism is the same as the FP32 original: each thread copies an
// exclusive row range, identified by params.ith / params.nth.
func ComputeForwardDupI8(params *ComputeParams, src0, dst *Tensor) {

	blockSize := BLCK_SIZE[src0.Type] // Q8_0 block size

	if !dst.IsContiguous() {
		fmt.Println(dst.Type)
		fmt.Println(dst.NB[0], dst.NB[1], dst.NB[2], dst.NB[3])
		fmt.Println(blockSize, dst.Nelements())
		fmt.Fprintln(os.Stderr, "[HALT] ComputeForwardDupQ8: dst is NOT contiguous")
		os.Exit(1)
	}

	if src0.Type != TYPE_I8 || dst.Type != TYPE_I8 {
		fmt.Fprintln(os.Stderr, "[HALT] ComputeForwardDupQ8: tensors must be TYPE_Q8_0")
		os.Exit(1)
	}
	if src0.Nelements() != dst.Nelements() {
		fmt.Fprintln(os.Stderr, "[HALT] ComputeForwardDupQ8: element counts differ")
		os.Exit(1)
	}
	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	// ─────────────────────────────────────────────────────────────
	// Fast path: both tensors are contiguous → just memcpy both blobs
	// ─────────────────────────────────────────────────────────────
	if src0.IsContiguous() && dst.IsContiguous() {
		copy(dst.Data, src0.Data)
		copy(dst.Scalars, src0.Scalars)
		return
	}

	// ─────────────────────────────────────────────────────────────
	// Generic (non-contiguous) path
	// ─────────────────────────────────────────────────────────────

	ne00 := src0.NE[0]
	ne01 := src0.NE[1]
	ne02 := src0.NE[2]
	ne03 := src0.NE[3]

	nb00 := src0.NB[0] // byte-stride for one i00 step
	nb01 := src0.NB[1]
	nb02 := src0.NB[2]
	nb03 := src0.NB[3]

	// Work split per thread = product of the three outer dims
	totalRows := ne01 * ne02 * ne03
	rowsPerTh := (totalRows + params.nth - 1) / 1 //params.nth
	rowStart := rowsPerTh * params.ith
	rowEnd := min32(rowStart+rowsPerTh, totalRows)

	// Each outer-dimension index triple (i01,i02,i03) defines one “row”.
	// We convert row index ↦ 3-D indices with the same arithmetic the FP32
	// version used.
	for row := rowStart; row < rowEnd; row++ {

		i03 := row / (ne01 * ne02)
		rest := row - i03*ne01*ne02
		i02 := rest / ne01
		i01 := rest - i02*ne01

		// Base byte offsets of this row in src and dst
		srcRowOff := i01*nb01 + i02*nb02 + i03*nb03
		dstRowOff := srcRowOff // identical layout in dst

		// Base block index of this row in the scalar slice
		// (32 int8 values = 1 block)
		blocksBeforeRow := (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00) / blockSize

		for i00 := uint32(0); i00 < ne00; i00 += blockSize {

			// --- copy 32 int8 values -----------------------------------------
			srcBlockOff := srcRowOff + i00*nb00
			dstBlockOff := dstRowOff + i00*nb00
			copy(dst.Data[dstBlockOff:dstBlockOff+blockSize], src0.Data[srcBlockOff:srcBlockOff+blockSize])

			// --- copy 1 float16 scale ----------------------------------------
			blkIdx := blocksBeforeRow + i00/blockSize
			dst.Scalars[blkIdx] = src0.Scalars[blkIdx]
		}
	}
}

// ggml_compute_forward_reshape
func ComputeForwardReshape(params *ComputeParams, src0, dst *Tensor) {
	// NOP
}

// ggml_compute_forward_permute
func ComputeForwardPermute(params *ComputeParams, src0 *Tensor) {
	// NOP
}

// ComputeForwardRopeI8 applies rotary position embeddings to the input tensor using int8 quantization.
// It dequantizes the input, performs the rotation in float32, and then re-quantizes the output.
func ComputeForwardRopeI8(params *ComputeParams, src0, src1, dst *Tensor) {

	////assert(params->ith == 0);
	////assert(src1->type == GGML_TYPE_I32);
	////assert(ggml_nelements(src1) == 3);

	if src1.Nelements() != 3 {
		fmt.Printf("\n[HALT] ComputeForwardRopeFP32 : src1 has NOT EXACT 3 elements!")
		os.Exit(1)
	}

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	n_past := int(src1.Data[0])
	dims := uint32(src1.Data[1])
	n_tokens := int(src0.NE[1])
	n_embd := int(src0.NE[0])

	// These would typically come from the model's hyperparameters
	rope_freq_base := float32(10000.0)
	rope_freq_scale := float32(1.0)

	for it := 0; it < n_tokens; it++ {
		pos := n_past + it

		// Temporary buffer for one row of float32 results
		dst_row_f32 := make([]float16.Float16, n_embd)

		// Apply RoPE to the first n_rot dimensions
		for i := 0; i < int(dims); i += 2 {
			// Dequantize the input pair from src0
			idx0 := it*n_embd + i
			idx1 := it*n_embd + i + 1

			scalar0 := src0.Scalars[idx0/32]
			scalar1 := src0.Scalars[idx1/32]

			x0_f32 := float16.Float16(src0.Data[idx0]) * scalar0
			x1_f32 := float16.Float16(src0.Data[idx1]) * scalar1

			// Calculate sin and cos for the current position
			inv_freq := 1.0 / (math.Pow(float64(rope_freq_base), float64(i)/float64(dims)) * float64(rope_freq_scale))
			freq := float32(1.0 / inv_freq)
			sin_val := float16.Float16(math.Sin(float64(float32(pos) * freq)))
			cos_val := float16.Float16(math.Cos(float64(float32(pos) * freq)))

			// Apply the 2D rotation
			dst_row_f32[i] = x0_f32*cos_val - x1_f32*sin_val
			dst_row_f32[i+1] = x0_f32*sin_val + x1_f32*cos_val
		}

		// Dequantize the remaining dimensions that are not rotated
		for i := int(dims); i < n_embd; i++ {
			idx := it*n_embd + i
			scalar := src0.Scalars[idx/32]
			dst_row_f32[i] = float16.Float16(src0.Data[idx]) * scalar
		}

		// Re-quantize the float32 row back to int8 blocks for the destination tensor
		for i := 0; i < n_embd; i += 32 {
			block_f32 := dst_row_f32[i : i+32]

			// Find the maximum absolute value in the block to determine the scaling factor
			max_abs_val := float16.Float16(0.0)
			for _, val := range block_f32 {
				abs_val := float16.Float16(math.Abs(float64(val)))
				if abs_val > max_abs_val {
					max_abs_val = abs_val
				}
			}

			// Define the new scalar for this block
			new_scalar := max_abs_val / 127
			if new_scalar == 0.0 {
				new_scalar = 1.0
			}

			dst_scalar_idx := (it*n_embd + i) / 32
			dst.Scalars[dst_scalar_idx] = new_scalar

			// Quantize the block and store it in the destination tensor's data
			for j, val := range block_f32 {
				dst_idx := it*n_embd + i + j
				dst.Data[dst_idx] = int8(val / new_scalar)
			}
		}
	}
}

// ggml_compute_forward_scale_f32
func ComputeForwardScaleI8(params *ComputeParams, src0, src1, dst *Tensor) {

	////GGML_ASSERT(ggml_is_contiguous(src0));
	////GGML_ASSERT(ggml_is_contiguous(dst));
	////GGML_ASSERT(ggml_are_same_shape(src0, dst));
	////GGML_ASSERT(ggml_is_scalar(src1));

	if !src0.IsContiguous() {
		fmt.Printf("[HALT] ComputeForwardScaleFP32 : [src0] is NOT contiguous!")
		os.Exit(1)
	}

	if !dst.IsContiguous() {
		fmt.Printf("[HALT] ComputeForwardScaleFP32 : [dst] is NOT contiguous!")
		os.Exit(1)
	}

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	// scale factor
	sv := src1.Scalars[0]
	v := src1.Data[0]

	ith := params.ith
	nth := params.nth

	//nc := src0.NE[0]
	//nr := src0.Nrows()
	ne := src0.Nelements()

	// chunks per thread
	nBlocks := uint32(math.Ceil(float64(ne) / 32))
	blocksPerThread := uint32(math.Ceil(float64(nBlocks) / float64(nth)))

	//// row range for this thread
	//ir0 := dr * ith
	//ir1 := min(int(ir0)+int(dr), int(nr))

	for c := uint32(ith * blocksPerThread); c < ((ith + 1) * blocksPerThread); c++ {
		////ggml_vec_scale_f32(nc, (float *) ((char *) dst->data + i1*(dst->nb[1])), v);
		////VecScaleFP32(nc, (*dst.Data)[i1*dst.NE[0]:], v)
		VecScaleI8(dst.Scalars[c], sv*float16.Float16(v))
	}

}

// ggml_compute_forward_diag_mask_inf

// ComputeForwardDiagMaskInfI8 applies a diagonal mask to an int8 tensor.
// It copies the src0 tensor to dst, then sets the upper triangular part of
// each matrix to math.MinInt8. This value should be interpreted as -infinity
// in subsequent operations like softmax.
func ComputeForwardDiagMaskInfI8(params *ComputeParams, src0, src1, dst *Tensor) {
	// Step 1: Copy src0 to dst. This ensures that the non-masked elements are preserved,
	// effectively making this an out-of-place operation.
	copy(dst.Data, src0.Data)
	copy(dst.Scalars, src0.Scalars)
	dst.NE = src0.NE
	dst.NB = src0.NB
	dst.Type = src0.Type
	dst.Dims = src0.Dims

	// Step 2: Dequantize pastCount from the src1 tensor.
	// Since src1 is a quantized tensor with a single value, we dequantize it
	// by multiplying its int8 data value with its float32 scalar.
	pastCountF32 := float32(float16.Float16(src1.Data[0]) * src1.Scalars[0])

	// We round the result to the nearest integer to get the final pastCount.
	pastCount := uint32(pastCountF32 + 0.5)

	// Step 3: Apply the mask.
	nc := src0.NE[0]
	nr := src0.NE[1]

	// Calculate the number of matrices (batches).
	var nz uint32 = 1
	if src0.Dims > 2 {
		nz = src0.Nrows() / nr
	}

	// Iterate and apply the mask. The strides (NB) are in bytes. Since the
	// data type is int8 (1 byte), the stride is equal to the index offset.
	for k := uint32(0); k < nz; k++ {
		for j := uint32(0); j < nr; j++ {
			// The mask is applied to the upper triangular part of the matrix,
			// starting from the diagonal offset by pastCount.
			for i := pastCount + j + 1; i < nc; i++ {
				idx := k*src0.NB[2] + j*src0.NB[1] + i*src0.NB[0]
				dst.Data[idx] = math.MinInt8
			}
		}
	}

}

func maxFloat(x, y float32) float32 {
	if x >= y {
		return x
	}
	return y
}

func VecMaxFP32(n uint32, x []float32) float32 {
	max := float32(math.Inf(-1)) // TODO use constant
	for i := uint32(0); i < n; i++ {
		max = maxFloat(max, x[i])
	}
	return max
}

// ggml_compute_forward_soft_max

// ComputeForwardSoftMaxI8 calculates the softmax of an int8 tensor and stores the result
// in another int8 tensor. It processes each row of the input tensor independently.
func ComputeForwardSoftMaxI8(params *ComputeParams, src0, dst *Tensor) {
	// Typically, softmax is applied on the last dimension.
	// We'll treat the tensor as a 2D matrix of (N, M) where N is the number of rows
	// and M is the number of elements in the last dimension.
	n_rows := int(src0.Nrows())
	n_cols := int(src0.NE[0])

	// Process one row at a time
	for i := 0; i < n_rows; i++ {
		// Temporary float32 slice to hold the dequantized row and the softmax result
		row_f32 := make([]float16.Float16, n_cols)

		// 1. Dequantize the current row from src0 to float32
		for j := 0; j < n_cols; j++ {
			idx := i*n_cols + j
			scalar := src0.Scalars[idx/32]
			row_f32[j] = float16.Float16(src0.Data[idx]) * scalar
		}

		// 2. Perform softmax on the float32 row
		// a. Find the maximum value in the row for numerical stability
		max_val := row_f32[0]
		for j := 1; j < n_cols; j++ {
			if row_f32[j] > max_val {
				max_val = row_f32[j]
			}
		}

		// b. Exponentiate and sum
		var sum_exp float16.Float16
		for j := 0; j < n_cols; j++ {
			// Subtract max_val before exponentiating
			row_f32[j] = float16.Float16(math.Exp(float64(row_f32[j] - max_val)))
			sum_exp += row_f32[j]
		}

		// c. Normalize
		for j := 0; j < n_cols; j++ {
			row_f32[j] /= sum_exp
		}

		// 3. Re-quantize the resulting float32 row into the dst tensor
		for j := 0; j < n_cols; j += 32 {
			end := j + 32
			if end > n_cols {
				end = n_cols
			}
			block_f32 := row_f32[j:end]

			// Find the maximum absolute value in the block to determine the scaling factor.
			// For softmax, the result is always positive, so we just need the max value.
			max_block_val := float16.Float16(0.0)
			for _, val := range block_f32 {
				if val > max_block_val {
					max_block_val = val
				}
			}

			// Calculate the new scalar for this block
			new_scalar := max_block_val / 127
			if new_scalar == 0.0 {
				new_scalar = 1.0 // Avoid division by zero
			}

			dst_scalar_idx := (i*n_cols + j) / 32
			dst.Scalars[dst_scalar_idx] = new_scalar

			// Quantize the block and store it in the destination tensor's data
			for k, val := range block_f32 {
				dst_idx := i*n_cols + j + k
				dst.Data[dst_idx] = int8(val / new_scalar)
			}
		}
	}
}

// inline static void ggml_vec_add_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i] + y[i]; }
func VecAddI8(scalarZ *float16.Float16, z []int8, scalarX float16.Float16, x []int8, scalarY float16.Float16, y []int8) {
	dstF32 := make([]float16.Float16, 32)

	for i := uint32(0); i < uint32(len(x)); i++ {
		dstF32[i] = (scalarX * float16.Float16(x[i])) + (scalarY * float16.Float16(y[i]))
	}

	maxValue := dstF32[0]
	for _, val := range dstF32 {
		if val > maxValue {
			maxValue = val
		}
	}

	*scalarZ = maxValue / 127

	for i, val := range dstF32 {
		z[i] = int8(val / *scalarZ)
	}
}

// ggml_compute_forward_add
func ComputeForwardAddI8(params *ComputeParams, src0, src1, dst *Tensor) {

	////GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	if src1.NB[0] != TYPE_SIZE[TYPE_I8] {
		fmt.Printf("[HALT] ComputeForwardAddI8 : [src1] is NOT contiguous!")
		os.Exit(1)
	}

	ith := params.ith
	nth := params.nth

	n := src0.Nrows()
	//nc := src0.NE[0]
	//
	////nb00 := src0.NB[0]
	//nb01 := src0.NB[1]
	//
	nb10 := src1.NB[0]
	//nb11 := src1.NB[1]
	//
	////nb0 := dst.NB[0]
	//nb1 := dst.NB[1]

	////GGML_ASSERT( nb0 == sizeof(float));
	////GGML_ASSERT(nb00 == sizeof(float));
	nChunks := (n / 32 / nth)
	j0 := nChunks * ith

	rl := src0.NE[0]

	if j0*32 > n {
		return
	} // Out-of-bounds check
	if nb10 == TYPE_SIZE[TYPE_I8] {

		//// j1 := ith == nth - 1 ? n : (n/nth)*(ith + 1)
		//var j1 uint32
		//if ith == nth-1 {
		//	j1 = n
		//} else {
		//	j1 = (n / nth) * (ith + 1)
		//}

		for c := j0; c < (j0 + nChunks); c++ {
			if (c + 1) > n/32 {
				continue
			}
			// Add the scalar data with /32 offset
			////ggml_vec_add_f32(nc,
			////        (float *) ((char *) dst->data  + j*nb1),
			////        (float *) ((char *) src0->data + j*nb01),
			////        (float *) ((char *) src1->data + j*nb11));

			VecAddI8(
				&dst.Scalars[c],
				dst.Data[c*32:(c+1)*32],
				src0.Scalars[c],
				src0.Data[c*32:(c+1)*32],
				src1.Scalars[c], // remove division by 4 for indexing?
				src1.Data[c*32:(c+1)*32],
			)
		}

	} else { // src1 is not contiguous

		for c := j0; c < (j0 + nChunks); c++ {
			if (c + 1) > n/32 {
				continue
			}
			// Add the scalar data with /32 offset
			////ggml_vec_add_f32(nc,
			////        (float *) ((char *) dst->data  + j*nb1),
			////        (float *) ((char *) src0->data + j*nb01),
			////        (float *) ((char *) src1->data + j*nb11));

			for i := uint32(0); i < uint32(32); i++ {
				tRow := ((c*32 + i) % rl)
				tCol := c*32 + i/rl

				VecAddI8(
					&dst.Scalars[c],
					dst.Data[c*32+i:c*32+i+1],
					src0.Scalars[c],
					src0.Data[c*32+i:c*32+i+1],
					src1.Scalars[tRow*rl+tCol], // remove division by 4 for indexing?
					src1.Data[tRow*rl+tCol:tRow*rl+tCol+1],
				)
			}
		}

		//for c := j0; c < (j0 + nChunks); c++ {
		//	if (c + 1) > n/32 {
		//		continue
		//	}
		//}
		//for j := ith; j < n; j += nth {
		//	////float * dst_ptr  = (float *) ((char *) dst->data  + j*nb1);
		//	dstPtr := dst.Data[j*nb1/4:]
		//	////float * src0_ptr = (float *) ((char *) src0->data + j*nb01);
		//	src0Ptr := src0.Data[j*nb01/4:]
		//	src0scl := src0.Scalars[j*nb01/4/32:]
		//	for i := uint32(0); i < nc; i++ {
		//		////float * src1_ptr = (float *) ((char *) src1->data + j*nb11 + i*nb10);
		//		src1Ptr := src1.Data[j*nb11/4+i*nb10/4]
		//		dstPtr[i] = src0Ptr[i] + src1Ptr
		//	}
		//}
	}

	if DEBUG {
		fmt.Printf("\n\n>>> OUT <<< ComputeForwardAddI8 <<<")
	}
}

// Sigmoid Linear Unit (SiLU) function
func SiluFP32(x float16.Float16) float16.Float16 {
	return x / float16.Float16(1.0+math.Exp(float64(-x)))
}

// inline static void ggml_vec_silu_f32(const int n, float * y, const float * x) {
func VecSiluI8(scalarY *float16.Float16, y []int8, scalarX float16.Float16, x []int8) {
	dstF32 := make([]float16.Float16, 32)
	maxValue := float16.Float16(0.0)

	for i := uint32(0); i < uint32(len(x)); i++ {
		dstF32[i] = SiluFP32(scalarX * float16.Float16(x[i])) // ggml_silu_f32

		if abs := float16.Float16(math.Abs(float64(dstF32[i]))); abs > maxValue {
			maxValue = abs
		}
	}

	*scalarY = maxValue / 127

	for i, val := range dstF32 {
		y[i] = int8(val / *scalarY)
	}
}

// ggml_compute_forward_silu
func ComputeForwardSiluI8(params *ComputeParams, src0, dst *Tensor) {

	////GGML_ASSERT(ggml_is_contiguous(src0));
	////GGML_ASSERT(ggml_is_contiguous(dst));
	////GGML_ASSERT(ggml_are_same_shape(src0, dst));

	if !src0.IsContiguous() {
		fmt.Printf("[HALT] ComputeForwardSiluFP32 : [src0] is NOT contiguous!")
		os.Exit(1)
	}

	if !dst.IsContiguous() {
		fmt.Printf("[HALT] ComputeForwardSiluFP32 : [dst] is NOT contiguous!")
		os.Exit(1)
	}

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	// Configure the destination tensor's metadata.
	dst.NE = src0.NE
	dst.NB = src0.NB
	dst.Type = TYPE_I8
	dst.op = OP_SILU

	// Determine the number of chunks to process. Tensors are processed in chunks of 32 elements.
	ne := dst.Nelements()
	nb := ne / 32

	// Distribute the chunks across the available threads for parallel processing.
	nth := params.nth
	ith := params.ith
	nChunksPerThread := (nb + nth - 1) / nth
	startChunk := ith * nChunksPerThread
	endChunk := (ith + 1) * nChunksPerThread
	if endChunk > nb {
		endChunk = nb
	}

	// Process the assigned chunks.
	for i := startChunk; i < endChunk; i++ {
		dataIdx := i * 32
		scalarIdx := i

		// Execute the vectorized SiLU operation for the current chunk.
		VecSiluI8(
			&dst.Scalars[scalarIdx],
			dst.Data[dataIdx:dataIdx+32],
			src0.Scalars[scalarIdx],
			src0.Data[dataIdx:dataIdx+32],
		)
	}

}

// ---

type TokenScore struct {
	Token string
	Score float32
}

type Vocab struct {
	Size     uint32
	Token2ID map[string]uint32
	ID2Token []TokenScore
}

func NewVocab(size uint32) *Vocab {
	return &Vocab{
		Token2ID: make(map[string]uint32, size),
		ID2Token: make([]TokenScore, size, size),
	}
}

func min(a, b int) int {
	if a <= b {
		return a
	}
	return b
}

// ---- SentencePiece Tokenizer

// struct llama_sp_symbol {
type Symbol struct {
	////using index = int;

	// NB! Allow -1
	Prev int
	Next int

	Text string
	N    uint32
}

// struct llama_sp_bigram {
type Bigram struct {

	// NB! Allow -1
	Left  int
	Right int

	Score float32
	Size  uint32
}

func utf8Len(src byte) uint32 {
	lookup := []uint32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4}
	highbits := uint8(src) >> 4
	return lookup[highbits]
}

func Token2Str(vocab *Vocab, token uint32) string {
	if int(token) >= len(vocab.ID2Token) {
		return ""
	}

	return vocab.ID2Token[token].Token
}

func PopMax(queue *[]Bigram) Bigram {

	max := 0 // index of max score element in queue
	for cur := 1; cur < len(*queue); cur++ {
		if ((*queue)[max].Score < (*queue)[cur].Score) ||
			((*queue)[max].Score == (*queue)[cur].Score &&
				(*queue)[max].Left > (*queue)[cur].Left) {
			max = cur
		}
	}

	pop := (*queue)[max]

	// replace max element with last and shrink slice (if max == last, then just remove it)
	(*queue)[max] = (*queue)[len(*queue)-1]
	*queue = (*queue)[:len(*queue)-1]

	return pop
}

func TryAddBigram(vocab *Vocab, symbols []Symbol, workQueue *[]Bigram, left, right int) {

	if left == -1 || right == -1 {
		return
	}

	token := symbols[left].Text[:symbols[left].N+symbols[right].N]
	id, ok := vocab.Token2ID[token]

	if !ok || int(id) >= len(vocab.ID2Token) {
		return
	}

	tokenScore := vocab.ID2Token[id]

	bigram := Bigram{Left: left, Right: right, Score: tokenScore.Score, Size: uint32(len(token))}
	*workQueue = append(*workQueue, bigram)
}

const NewLineToken = 13 // ml.Tokenize(Ctx.Vocab, "\n", false)[0]

// void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
func Tokenize(vocab *Vocab, text string, bos bool) []uint32 {

	output := make([]uint32, 0)
	symbols := make([]Symbol, 0)   // std::vector<llama_sp_symbol> symbols_;
	workQueue := make([]Bigram, 0) // llama_sp_bigram::queue work_queue_; // std::priority_queue<llama_sp_bigram, queue_storage, comparator>;

	if bos {
		output = append(output, 1) // TODO: replace with vocab.bos
	}

	// --- split string into utf8 chars

	index := 0
	offs := 0
	for offs < len(text) {
		var sym Symbol
		charLen := min(len(text)-offs, int(utf8Len(text[offs])))
		sym.Text = text[offs:]
		sym.N = uint32(charLen)
		offs += charLen
		sym.Prev = index - 1
		if offs == len(text) {
			sym.Next = -1
		} else {
			sym.Next = index + 1
		}
		index++
		symbols = append(symbols, sym)
	}

	// seed the work queue with all possible 2-character tokens
	for i := 1; i < len(symbols); i++ {
		TryAddBigram(vocab, symbols, &workQueue, i-1, i)
	}

	// keep substituting the highest frequency pairs for as long as we can
	for len(workQueue) > 0 {
		bigram := PopMax(&workQueue)

		leftSym := &symbols[bigram.Left]
		rightSym := &symbols[bigram.Right]

		// if one of the symbols already got merged, skip it
		if leftSym.N == 0 || rightSym.N == 0 || leftSym.N+rightSym.N != bigram.Size {
			continue
		}

		// merge the right sym into the left one
		leftSym.N += rightSym.N
		rightSym.N = 0

		// remove the right sym from the chain
		leftSym.Next = rightSym.Next
		if rightSym.Next >= 0 {
			symbols[rightSym.Next].Prev = bigram.Left
		}

		// find more substitutions
		TryAddBigram(vocab, symbols, &workQueue, leftSym.Prev, bigram.Left)
		TryAddBigram(vocab, symbols, &workQueue, bigram.Left, leftSym.Next)
	}

	for i := 0; i != -1; i = symbols[i].Next {
		symbol := symbols[i]
		id, ok := vocab.Token2ID[symbol.Text[:symbol.N]]

		if !ok {
			// output any symbols that did not form tokens as bytes.
			for j := uint32(0); j < symbol.N; j++ {
				////llama_vocab::id token_id = static_cast<uint8_t>(symbol.text[j]) + 3;
				tokenID := uint32(symbol.Text[j] + 3)
				output = append(output, tokenID)
			}
		} else {
			output = append(output, id)
		}
	}

	if DEBUG {
		fmt.Printf("\n\n=== TOKENIZER ===\n\n%+v", output)
		for i := 0; i < len(output); i++ {
			fmt.Printf("%d:'%s'  ", output[i], Token2Str(vocab, output[i]))
		}
	}

	return output

}

// TODO Do we need this?
func Init(params InitParams) {

	// ---- initialize GELU, SILU and EXP F32 tables

	////const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

	/////////////////////////////////////////var ii uint16
	/////////////////////////////////////////for i := uint32(0); i < (1 << 16); i++ {
	/////////////////////////////////////////ui := uint16(i)

	////memcpy(&ii, &ui, sizeof(ii));
	////const float f = table_f32_f16[i] = COMPUTE_FP16_TO_FP32(ii);
	/////////////////////////////////////////fp32 := float32()

	////table_gelu_f16[i] = FP32_TO_FP16(ggml_gelu_f32(f));
	////table_silu_f16[i] = FP32_TO_FP16(ggml_silu_f32(f));

	////TableExpFP16[i]  = FP32_TO_FP16(exp(f));
	/////////////////////////////////////////exp := float32(math.Exp(fp32))
	/////////////////////////////////////////TableExpFP16[i] = float16.Fromfloat32(exp)

	/////////////////////////////////////////}

	////const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

}

// Allocator is an experimental memory pool for FP32 slices
// TODO: Investigate https://github.com/valyala/bytebufferpool
type Allocator struct {
	sync.Mutex

	// TODO: [][]float32 vs []*[]float32
	// Used map[uint32][]*[]float32
	// Free map[uint32][]*[]float32

	PoolSize int
	MemSize  int

	Pool []byte
	Mem  []byte
}

// TODO: Precompute max needed RAM size
const MaxPool = 0 // 2_000_000_000
const MaxMem = 0  // 28_000_000_000

func NewAllocator() *Allocator {
	return &Allocator{
		// Used: make(map[uint32][]*[]float32),
		// Free: make(map[uint32][]*[]float32),
		Pool: make([]byte, MaxPool),
		Mem:  make([]byte, MaxMem),
	}
}

// Get new or reuse memory buffer of size bytes
func (a *Allocator) Get(size uint32) *[]float32 {
	//gcSlice := make([]float32, size, size)
	//return &gcSlice

	a.Lock()
	byteSize := int(size * 4)

	if a.PoolSize+byteSize >= MaxPool {
		fmt.Printf("[ HALT ] Allocator go over free POOL MEM")
		os.Exit(0)
	}

	cur := a.PoolSize
	a.PoolSize += byteSize
	a.Unlock()

	var slice []float32
	head := (*reflect.SliceHeader)(unsafe.Pointer(&slice))
	head.Len = int(size)
	head.Cap = int(size)
	head.Data = uintptr(unsafe.Pointer(&a.Pool[cur]))

	return &slice

	/*
		head := reflect.SliceHeader{
			Len:  size,
			Cap:  size,
			Data: (*[]float32)(unsafe.Pointer(&a.Mem[cur])),
		}*/

	/*
	   _, ok := a.Free[size]

	   	if !ok {
	   		a.Used[size] = make([]*[]float32, 0, 1024) // Which CAP default?
	   		a.Free[size] = make([]*[]float32, 0, 1024) // Which CAP default?
	   	}

	   available := len(a.Free[size])

	   	if available > 0 {
	   		slice := a.Free[size][available-1]
	   		a.Free[size] = a.Free[size][:available-1]
	   		a.Used[size] = append(a.Used[size], slice)
	   		return slice
	   	}

	   ///slice := make([]float32, size, size)
	   a.Used[size] = append(a.Used[size], &slice)
	   return &slice
	*/
}

// Get fixed memory buffer of size bytes
func (a *Allocator) GetFixed(size uint32) *[]float32 {
	//gcSlice := make([]float32, size, size)
	//return &gcSlice

	a.Lock()
	byteSize := int(size * 4)

	if a.MemSize+byteSize >= MaxMem {
		fmt.Printf("[ HALT ] Allocator go over free FIXED MEM")
		os.Exit(0)
	}

	cur := a.MemSize
	a.MemSize += byteSize
	a.Unlock()

	var slice []float32
	head := (*reflect.SliceHeader)(unsafe.Pointer(&slice))
	head.Len = int(size)
	head.Cap = int(size)
	head.Data = uintptr(unsafe.Pointer(&a.Mem[cur]))

	return &slice

	/*
		head := reflect.SliceHeader{
			Len:  size,
			Cap:  size,
			Data: (*[]float32)(unsafe.Pointer(&a.Mem[cur])),
		}*/
}

func (a *Allocator) Reset() {
	a.Lock()
	a.PoolSize = 0
	a.Unlock()
	runtime.GC()

	// var rtm runtime.MemStats
	// runtime.ReadMemStats(&rtm)
	// printMemStats("Start", rtm)

	/*
	   	for size, _ := range a.Used {
	   		a.Free[size] = append(a.Free[size], a.Used[size]...)
	   		a.Used[size] = a.Used[size][:0]
	   	}

	   fmt.Printf("")
	*/
}

func printMemStats(message string, rtm runtime.MemStats) {
	fmt.Println("\n===", message, "===")
	fmt.Println("Mallocs: ", rtm.Mallocs)
	fmt.Println("Frees: ", rtm.Frees)
	fmt.Println("LiveObjects: ", rtm.Mallocs-rtm.Frees)
	fmt.Println("PauseTotalNs: ", rtm.PauseTotalNs)
	fmt.Println("NumGC: ", rtm.NumGC)
	fmt.Println("LastGC: ", time.UnixMilli(int64(rtm.LastGC/1_000_000)))
	fmt.Println("HeapObjects: ", rtm.HeapObjects)
	fmt.Println("HeapAlloc: ", rtm.HeapAlloc)
}

/*
// Release memory buffer back
func (a *Allocator) Put(size uint32, slice []float32) {

}

// Release memory buffer back
func (a Allocator) PutTensor(tensor *Tensor) {
	size := tensor.NE[0] * tensor.NE[1] * tensor.NE[2] * tensor.NE[3]
	_, ok := a.Pool[size]
	if !ok {
		a.Pool[size] = make([][]float32, 0, 64) // Which CAP default?
	}
	a.Pool[size] = append(a.Pool[size], tensor.Data)
	tensor.Data = nil
}
*/

/*
func NewReusableTensor1D(ctx *Context, dt DType, ne0 uint32) *Tensor {
	return NewTensor(ctx, dt, 1, ne0, 1, 1, 1, nil) // Reusable OK
}

func NewReusableTensor2D(ctx *Context, dt DType, ne0, ne1 uint32) *Tensor {
	return NewTensor(ctx, dt, 2, ne0, ne1, 1, 1, nil) // Reusable OK
}

func NewReusableTensor3D(ctx *Context, dt DType, ne0, ne1, ne2 uint32) *Tensor {
	return NewTensor(ctx, dt, 3, ne0, ne1, ne2, 1, nil) // Reusable OK
}

func NewFixedTensor1D(ctx *Context, dt DType, ne0 uint32) *Tensor {
	return NewFixedTensor(ctx, dt, 1, ne0, 1, 1, 1, nil)
}

func NewFixedTensor2D(ctx *Context, dt DType, ne0, ne1 uint32) *Tensor {
	return NewFixedTensor(ctx, dt, 2, ne0, ne1, 1, 1, nil)
}
*/
/*
// ggml_new_tensor_impl
func NewReusableTensor(ctx *Context, dt DType, dims uint32, ne0, ne1, ne2, ne3 uint32, data []float32) *Tensor {

	fmt.Printf("NewReusableTensor")
	os.Exit(1)

	// Reusable OK
	if data == nil {
		data = *ctx.Allocator.Get(ne0 * ne1 * ne2 * ne3)
	}

	//if data == nil {
	//	total := ne0 * ne1 * ne2 * ne3
	//	data = make([]float32, total, total)
	//}

	return &Tensor{
		Type:     dt,
		Reusable: true,
		Dims:     dims,
		NE:       [4]uint32{ne0, ne1, ne2, ne3},
		NB:       [4]uint32{4, ne0 * 4, ne0 * ne1 * 4, ne0 * ne1 * ne2 * 4},
		op:       OP_NONE,
		Data:     data,
	}
}
*/
/*
// ggml_new_tensor_impl
func NewFixedTensor(ctx *Context, dt DType, dims uint32, ne0, ne1, ne2, ne3 uint32, data []float32) *Tensor {

	fmt.Printf("NewFixedTensor")
	os.Exit(1)

	// TODO: Check allowed data types on graph creation
	//if dt != TYPE_F32 && dt != TYPE_I32 {
	//	fmt.Printf("\n[ERROR] NewTensorImpl got not supported type : %d", dt)
	//	os.Exit(1)
	//}

	////ggml_assert_aligned(result);

	if data == nil {
		total := ne0 * ne1 * ne2 * ne3
		data = make([]float32, total, total)

		// Reusable OK ???
		// data = *ctx.Allocator.GetFixed(ne0 * ne1 * ne2 * ne3)
	}

	return &Tensor{
		Type: dt,
		Dims: dims,
		NE:   [4]uint32{ne0, ne1, ne2, ne3},
		NB:   [4]uint32{4, ne0 * 4, ne0 * ne1 * 4, ne0 * ne1 * ne2 * 4},
		op:   OP_NONE,
		Data: data,
	}
}
*/
