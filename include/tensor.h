/* tensor.h -- dense N-D float tensor with a reverse-mode autograd engine.
 *
 * Design notes (miniTorch)
 * ------------------------
 * A Tensor is both a data holder and a node in a dynamic computation graph.
 * Every operation (add, matmul, relu, conv2d, ...) builds a new node that
 * remembers its parents and whatever it needs to differentiate itself.
 *
 * The graph is owned by the *outputs*: a node owns its parents. So calling
 * tensor_free(loss) after tensor_backward(loss) tears down the whole forward
 * pass of one iteration. Parameters are roots (no parents) and are therefore
 * never freed from inside the graph -- the caller owns them via a Scope.
 *
 * Gradients accumulate on the parameter tensors themselves, which is what
 * PyTorch's AccumulateGrad node does.
 */
#ifndef MINITORCH_TENSOR_H
#define MINITORCH_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Scope: registry for tensors that outlive a single forward graph      */
/* ------------------------------------------------------------------ */

typedef struct Tensor Tensor;
typedef struct MiniTorchScope MiniTorchScope;

MiniTorchScope *scope_create(void);
/* Frees every tensor still registered in the scope, then the scope. */
void scope_destroy(MiniTorchScope *s);
/* Tensors still alive in the scope (a leak counter for tests/demos). */
size_t scope_live_tensors(MiniTorchScope *s);
/* Head of the scope's tensor list, for diagnostics/leak reporting. */
Tensor *scope_first(MiniTorchScope *s);
/* Next entry in the scope's list, and the reference count of a tensor.
 * Both exist purely so tests can report leaks without reaching into structs. */
Tensor *tensor_scope_next(const Tensor *t);
int     tensor_refs(const Tensor *t);

/* ------------------------------------------------------------------ */
/* Tensor                                                              */
/* ------------------------------------------------------------------ */

typedef enum {
    OP_LEAF = 0,
    OP_ACCUM_GRAD,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_NEG,
    OP_MATMUL,
    OP_MATMUL_TN,
    OP_SUM,
    OP_SUM_DIM,
    OP_MEAN_DIM,
    OP_RESHAPE,
    OP_TRANSPOSE,
    OP_PERMUTE,
    OP_BROADCAST,
    OP_RELU,
    OP_SIGMOID,
    OP_TANH,
    OP_GELU,
    OP_EXP,
    OP_LOG,
    OP_SQRT,
    OP_POW,
    OP_MAXIMUM,
    OP_WHERE,
    OP_SOFTMAX,
    OP_CROSS_ENTROPY,
    OP_IM2COL,
    OP_MAXPOOL,
    OP_DROPOUT,
    OP_EMBEDDING,
    OP_ADD_BIAS,
    OP_COUNT
} OpKind;

struct Tensor {
    /* --- data --- */
    float  *data;
    float  *grad;          /* same element count as data, NULL until needed */
    int     ndim;
    int64_t shape[4];
    int64_t stride[4];     /* C-contiguous strides, zeros for broadcast axes */
    int64_t numel;

    /* --- autograd bookkeeping --- */
    int     requires_grad;
    OpKind  op;
    Tensor *parent[2];
    int     nparents;
    int     refs;          /* how many graph nodes hold this as a parent */
    Tensor *saved;         /* extra operand needed by the backward pass */
    void   *ctx;           /* op-specific payload owned by this node */
    float   scalar;        /* op-specific scalar (pow exponent, dim, reduction) */

    /* --- traversal flags --- */
    unsigned char mark;    /* scratch flag for DFS during backward */
    unsigned char freed;   /* tensor_free already ran */
    unsigned char is_leaf;
    unsigned char held;    /* a caller-visible handle owns this tensor */
    Tensor *next_in_scope;
    MiniTorchScope *scope;
    int     registered;
};

/* ---- construction ---- */
Tensor *tensor_create(MiniTorchScope *s, const int64_t *shape, int ndim,
                      const float *data, int requires_grad);
Tensor *tensor_zeros(MiniTorchScope *s, const int64_t *shape, int ndim);
Tensor *tensor_full(MiniTorchScope *s, const int64_t *shape, int ndim, float v);
Tensor *tensor_constant(MiniTorchScope *s, const int64_t *shape, int ndim, float v);
Tensor *tensor_clone(const Tensor *t);
Tensor *tensor_rand_uniform(MiniTorchScope *s, const int64_t *shape, int ndim,
                            float lo, float hi, unsigned long long *rng_state);
Tensor *tensor_rand_normal(MiniTorchScope *s, const int64_t *shape, int ndim,
                           float std, unsigned long long *rng_state);

/* Free this node and, transitively, every parent it owns. */
void tensor_free(Tensor *t);

/* ---- introspection ---- */
int     tensor_ndim(const Tensor *t);
int64_t tensor_size(const Tensor *t, int dim);
int64_t tensor_numel(const Tensor *t);
const int64_t *tensor_shape(const Tensor *t);
float  *tensor_data(Tensor *t);
float  *tensor_grad(Tensor *t);
int     tensor_requires_grad(const Tensor *t);
int     tensor_opcode(const Tensor *t);
const char *tensor_op_name(const Tensor *t);
void    tensor_op_rule(int op, const char **name, void **bwd1, void **bwd2);
int     tensor_same_shape(const Tensor *a, const Tensor *b);
void    tensor_zero_grad(Tensor *t);

float   tensor_get(const Tensor *t, const int64_t *idx);
void    tensor_set(Tensor *t, const int64_t *idx, float v);
void    tensor_print(const Tensor *t, const char *label);

/* ---- elementwise ---- */
Tensor *tensor_add(Tensor *a, Tensor *b);
Tensor *tensor_sub(Tensor *a, Tensor *b);
Tensor *tensor_mul(Tensor *a, Tensor *b);
Tensor *tensor_div(Tensor *a, Tensor *b);
Tensor *tensor_add_scalar(Tensor *a, float s);
Tensor *tensor_mul_scalar(Tensor *a, float s);
Tensor *tensor_neg(Tensor *a);
Tensor *tensor_maximum(Tensor *a, Tensor *b);
Tensor *tensor_where(const Tensor *cond, Tensor *a, Tensor *b);
Tensor *tensor_relu(Tensor *a);
Tensor *tensor_sigmoid(Tensor *a);
Tensor *tensor_tanh(Tensor *a);
Tensor *tensor_gelu(Tensor *a);           /* tanh approximation */
Tensor *tensor_exp(Tensor *a);
Tensor *tensor_log(Tensor *a);
Tensor *tensor_sqrt(Tensor *a);
Tensor *tensor_pow(Tensor *a, float p);

/* ---- linear algebra / shape ---- */
Tensor *tensor_matmul(Tensor *a, Tensor *b);       /* [M,K] x [K,N] -> [M,N] */
Tensor *tensor_matmul_tn(Tensor *a, Tensor *b);    /* [M,K] x [N,K] -> [M,N] */
Tensor *tensor_sum(Tensor *a);                     /* full reduce -> scalar */
Tensor *tensor_sum_dim(Tensor *a, int dim, int keepdim);
Tensor *tensor_mean_dim(Tensor *a, int dim, int keepdim);
Tensor *tensor_reshape(Tensor *a, const int64_t *shape, int ndim);
Tensor *tensor_transpose2d(Tensor *a);
Tensor *tensor_broadcast_to(Tensor *a, const int64_t *shape, int ndim);

/* ---- neural-network grade ops with hand-written backward ---- */
Tensor *tensor_softmax(Tensor *x);                                  /* over last dim */
Tensor *tensor_cross_entropy(Tensor *logits, Tensor *targets, int reduction); /* 1=mean 0=sum */
Tensor *tensor_conv2d(Tensor *x, Tensor *w, Tensor *b,
                      int stride, int padding, int dilation);       /* NCHW */
Tensor *tensor_im2col(Tensor *x, int k, int stride, int pad, int dilation);
Tensor *tensor_add_bias(Tensor *x, Tensor *b);                      /* [N,C,H,W] + [C] */
Tensor *tensor_maxpool2d(Tensor *x, int k, int stride);
Tensor *tensor_embedding(Tensor *weight, const int64_t *idx, int64_t n_idx);
Tensor *tensor_dropout(Tensor *x, float p, int training, unsigned long long *rng);

/* ---- autograd ---- */
void tensor_backward(Tensor *out, const float *seed); /* NULL seed => ones */
float tensor_numeric_grad(Tensor *param, const int64_t *idx, float eps,
                          Tensor *(*f)(Tensor *param, void *ud), void *ud);

/* ---- rng (shared with nn.c) ---- */
unsigned long long mt_rand_next(unsigned long long *s);
float mt_rand_uniform(unsigned long long *s);
/* Process-wide RNG stream. data.c installs the accessor; passing NULL as the
 * rng_state of tensor_rand_* uses it. */
extern unsigned long long *(*mt_default_rng)(void);

#ifdef __cplusplus
}
#endif

#endif /* MINITORCH_TENSOR_H */
