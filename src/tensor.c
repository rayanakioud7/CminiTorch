/* tensor.c -- dense N-D float tensor + reverse-mode autograd engine.
 *
 * Every differentiable thing in miniTorch funnels through the small "tape" in
 * this file: each op builds a node, records its parents and whatever it needs
 * for the backward pass, and gets a backward rule from the table below.
 *
 * Ownership model
 * ---------------
 * A node OWNS its parents. tensor_free(loss) therefore tears down the whole
 * forward graph of one iteration. Parameters are leaves (no parents), so they
 * are never freed from inside a graph -- the caller owns them through a
 * MiniTorchScope.
 */
#include "tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TF_MAXDIM 4

/* ================================================================== */
/* Scope                                                              */
/* ================================================================== */

struct MiniTorchScope {
    Tensor *head;
    size_t  live;
};

static void tf_free(Tensor *t);

/* ------------------------------------------------------------------ */
/* Op registry. The rules themselves live further down; only the       */
/* signatures are needed here so that tf_free can report op names.     */
/* ------------------------------------------------------------------ */

typedef struct {
    OpKind      kind;
    const char *name;
    void (*bwd1)(Tensor *, int, float *, const float *);
    void (*bwd2)(Tensor *, float *, float *, const float *);
} OpSpec;

static void bwd_neg(Tensor *, int, float *, const float *);
static void bwd_relu(Tensor *, int, float *, const float *);
static void bwd_sigmoid(Tensor *, int, float *, const float *);
static void bwd_tanh(Tensor *, int, float *, const float *);
static void bwd_gelu(Tensor *, int, float *, const float *);
static void bwd_exp(Tensor *, int, float *, const float *);
static void bwd_log(Tensor *, int, float *, const float *);
static void bwd_sqrt(Tensor *, int, float *, const float *);
static void bwd_pow(Tensor *, int, float *, const float *);
static void bwd_reshape(Tensor *, int, float *, const float *);
static void bwd_id(Tensor *, int, float *, const float *);
static void bwd_full_reduce(Tensor *, int, float *, const float *);
static void bwd_sum_dim(Tensor *, int, float *, const float *);
static void bwd_mean_dim(Tensor *, int, float *, const float *);
static void bwd_transpose(Tensor *, int, float *, const float *);
static void bwd_permute(Tensor *, int, float *, const float *);
static void bwd_im2col(Tensor *, int, float *, const float *);
static void bwd_maxpool(Tensor *, int, float *, const float *);
static void bwd_embedding(Tensor *, int, float *, const float *);
static void bwd_dropout(Tensor *, int, float *, const float *);
static void bwd_softmax(Tensor *, int, float *, const float *);
static void bwd_cross_entropy(Tensor *, int, float *, const float *);

static void bwd_add(Tensor *, float *, float *, const float *);
static void bwd_sub(Tensor *, float *, float *, const float *);
static void bwd_mul(Tensor *, float *, float *, const float *);
static void bwd_div(Tensor *, float *, float *, const float *);
static void bwd_maximum(Tensor *, float *, float *, const float *);
static void bwd_where(Tensor *, float *, float *, const float *);
static void bwd_matmul(Tensor *, float *, float *, const float *);
static void bwd_matmul_tn(Tensor *, float *, float *, const float *);
static void bwd_add_bias(Tensor *, float *, float *, const float *);

static const OpSpec g_ops[OP_COUNT] = {
    [OP_LEAF]          = { OP_LEAF,          "leaf",          NULL, NULL },
    [OP_ACCUM_GRAD]    = { OP_ACCUM_GRAD,    "accum_grad",    NULL, NULL },
    [OP_ADD]           = { OP_ADD,           "add",           NULL, bwd_add },
    [OP_SUB]           = { OP_SUB,           "sub",           NULL, bwd_sub },
    [OP_MUL]           = { OP_MUL,           "mul",           NULL, bwd_mul },
    [OP_DIV]           = { OP_DIV,           "div",           NULL, bwd_div },
    [OP_NEG]           = { OP_NEG,           "neg",           bwd_neg, NULL },
    [OP_MATMUL]        = { OP_MATMUL,        "matmul",        NULL, bwd_matmul },
    [OP_MATMUL_TN]     = { OP_MATMUL_TN,     "matmul_tn",     NULL, bwd_matmul_tn },
    [OP_SUM]           = { OP_SUM,           "sum",           bwd_full_reduce, NULL },
    [OP_SUM_DIM]       = { OP_SUM_DIM,       "sum_dim",       bwd_sum_dim, NULL },
    [OP_MEAN_DIM]      = { OP_MEAN_DIM,      "mean_dim",      bwd_mean_dim, NULL },
    [OP_RESHAPE]       = { OP_RESHAPE,       "reshape",       bwd_reshape, NULL },
    [OP_TRANSPOSE]     = { OP_TRANSPOSE,     "transpose",     bwd_transpose, NULL },
    [OP_PERMUTE]       = { OP_PERMUTE,       "permute",       bwd_permute, NULL },
    [OP_BROADCAST]     = { OP_BROADCAST,     "broadcast",     bwd_id, NULL },
    [OP_RELU]          = { OP_RELU,          "relu",          bwd_relu, NULL },
    [OP_SIGMOID]       = { OP_SIGMOID,       "sigmoid",       bwd_sigmoid, NULL },
    [OP_TANH]          = { OP_TANH,          "tanh",          bwd_tanh, NULL },
    [OP_GELU]          = { OP_GELU,          "gelu",          bwd_gelu, NULL },
    [OP_EXP]           = { OP_EXP,           "exp",           bwd_exp, NULL },
    [OP_LOG]           = { OP_LOG,           "log",           bwd_log, NULL },
    [OP_SQRT]          = { OP_SQRT,          "sqrt",          bwd_sqrt, NULL },
    [OP_POW]           = { OP_POW,           "pow",           bwd_pow, NULL },
    [OP_MAXIMUM]       = { OP_MAXIMUM,       "maximum",       NULL, bwd_maximum },
    [OP_WHERE]         = { OP_WHERE,         "where",         NULL, bwd_where },
    [OP_SOFTMAX]       = { OP_SOFTMAX,       "softmax",       bwd_softmax, NULL },
    [OP_CROSS_ENTROPY] = { OP_CROSS_ENTROPY, "cross_entropy", bwd_cross_entropy, NULL },
    [OP_IM2COL]        = { OP_IM2COL,        "im2col",        bwd_im2col, NULL },
    [OP_MAXPOOL]       = { OP_MAXPOOL,       "maxpool",       bwd_maxpool, NULL },
    [OP_DROPOUT]       = { OP_DROPOUT,       "dropout",       bwd_dropout, NULL },
    [OP_EMBEDDING]     = { OP_EMBEDDING,     "embedding",     bwd_embedding, NULL },
    [OP_ADD_BIAS]      = { OP_ADD_BIAS,      "add_bias",      NULL, bwd_add_bias },
};

MiniTorchScope *scope_create(void) { return (MiniTorchScope *)calloc(1, sizeof(MiniTorchScope)); }

size_t scope_live_tensors(MiniTorchScope *s) { return s ? s->live : 0; }

Tensor *scope_first(MiniTorchScope *s) { return s ? s->head : NULL; }

Tensor *tensor_scope_next(const Tensor *t) { return t ? t->next_in_scope : NULL; }

int tensor_refs(const Tensor *t) { return t ? t->refs : 0; }

static void scope_register(Tensor *t, MiniTorchScope *s) {
    if (!s) return;
    t->scope = s;
    t->next_in_scope = s->head;
    s->head = t;
    t->registered = 1;
    s->live++;
}

static void scope_unregister(Tensor *t) {
    MiniTorchScope *s = t->scope;
    if (!s || !t->registered) return;
    Tensor **pp = &s->head;
    while (*pp) {
        if (*pp == t) { *pp = t->next_in_scope; break; }
        pp = &(*pp)->next_in_scope;
    }
    t->registered = 0;
    t->next_in_scope = NULL;
    t->scope = NULL;
    if (s->live) s->live--;
}

void scope_destroy(MiniTorchScope *s) {
    if (!s) return;
    /* Detach the list first: tf_free unregisters and would mutate what we walk. */
    /* Pop from the head rather than caching a `next` pointer: freeing one
     * entry can pull other entries out of the same list, which would leave a
     * cached pointer dangling. */
    while (s->head) {
        Tensor *t = s->head;
        s->head = t->next_in_scope;
        t->next_in_scope = NULL;
        t->scope = NULL;
        t->registered = 0;
        if (s->live) s->live--;
        tf_free(t);
    }
    free(s);
}

/* ================================================================== */
/* Shape helpers                                                      */
/* ================================================================== */

static int64_t numel_of(const int64_t *shape, int ndim) {
    int64_t n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    return n;
}

static void contiguous_strides(const int64_t *shape, int ndim, int64_t *stride) {
    int64_t acc = 1;
    for (int i = ndim - 1; i >= 0; i--) { stride[i] = acc; acc *= shape[i]; }
    for (int i = ndim; i < TF_MAXDIM; i++) stride[i] = 0;
}

/* Coordinate of `flat` along dim `d` for a C-contiguous tensor of shape
 * `shape`/rank `ndim`. Callers must guarantee 0 <= d < ndim. */
static inline int64_t coord_of(int64_t flat, int d, const int64_t *shape, int ndim) {
    int64_t acc = 1;
    for (int i = d + 1; i < ndim; i++) acc *= shape[i];
    return acc > 0 ? (flat / acc) % shape[d] : 0;
}

/* Map output element `i` to a parent element, right-aligning axes so that
 * dropped (reduced), size-1 (broadcast) and expanded axes all behave. */
static int64_t map_to_parent(int64_t i, const Tensor *n, const Tensor *p) {
    int64_t off = 0;
    int shift = n->ndim - p->ndim;
    for (int pd = 0; pd < p->ndim; pd++) {
        int nd = pd + shift;
        int64_t c;
        if (nd < 0) c = 0;                                   /* axis absent -> 0 */
        else if (nd >= n->ndim) c = 0;                       /* extra leading   */
        else if (p->shape[pd] == 1) c = 0;                   /* broadcast axis  */
        else c = coord_of(i, nd, n->shape, n->ndim);
        off += c * p->stride[pd];
    }
    return off;
}

/* ================================================================== */
/* Node lifecycle                                                     */
/* ================================================================== */

static Tensor *tf_new(MiniTorchScope *s, const int64_t *shape, int ndim) {
    Tensor *t = (Tensor *)calloc(1, sizeof(Tensor));
    t->ndim = ndim > TF_MAXDIM ? TF_MAXDIM : ndim;
    for (int i = 0; i < t->ndim; i++) t->shape[i] = shape[i];
    contiguous_strides(t->shape, t->ndim, t->stride);
    t->numel = numel_of(t->shape, t->ndim);
    t->data = (float *)calloc((size_t)(t->numel > 0 ? t->numel : 1), sizeof(float));
    t->op = OP_LEAF;
    t->is_leaf = 1;
    scope_register(t, s);
    return t;
}

static void grad_ensure(Tensor *t) {
    if (!t->grad)
        t->grad = (float *)calloc((size_t)(t->numel > 0 ? t->numel : 1), sizeof(float));
}

static Tensor *tf_result(MiniTorchScope *s, const int64_t *shape, int ndim, OpKind op) {
    Tensor *t = tf_new(s, shape, ndim);
    t->op = op;
    t->is_leaf = 0;
    return t;
}

/* `saved` is a borrowed pointer for most ops (the operand is already a parent,
 * or it is a non-differentiable helper the caller owns). OP_WHERE is the
 * exception: it keeps the condition mask alive, so that edge is counted. */
static void tf_save(Tensor *node, Tensor *operand) {
    node->saved = operand;
    if (operand && node->op == OP_WHERE) operand->refs++;
}

static void tf_attach(Tensor *out, Tensor *p0, Tensor *p1) {
    out->parent[0] = p0;
    out->parent[1] = p1;
    out->nparents = p1 ? 2 : (p0 ? 1 : 0);
    out->requires_grad = 0;
    for (int i = 0; i < out->nparents; i++) {
        Tensor *p = out->parent[i];
        if (!p) continue;
        p->refs++;                     /* out now depends on p */
        if (p->requires_grad) out->requires_grad = 1;
    }
}

/* ================================================================== */
/* Free (iterative: transformer-depth graphs must not blow the stack)  */
/* ================================================================== */

static void tf_free_one(Tensor *t) {
    if (t->freed) return;
    t->freed = 1;
    free(t->data);
    free(t->grad);
    free(t->ctx);
    t->data = NULL;
    t->grad = NULL;
    t->ctx = NULL;
    scope_unregister(t);
    free(t);
}

/* Reference-counted teardown.
 *
 * Every edge (out -> parent) contributes one reference to the parent. Freeing
 * a node drops one reference from each of its parents and only frees a parent
 * once its count reaches zero. A tensor the caller still points at (refs == 0)
 * is therefore never released from inside a graph -- the old approach of
 * blindly walking parent edges used to free the parameters themselves.
 *
 * The traversal is iterative because a transformer-depth graph would otherwise
 * overflow the C stack.
 */
static void tf_free(Tensor *t) {
    if (!t) return;
    t->held = 0;
    size_t cap = 64, n = 0;
    Tensor **stack = (Tensor **)malloc(sizeof(Tensor *) * cap);

#define TF_PUSH(x)                                                              \
    do {                                                                        \
        if ((n + 1) >= cap) {                                                   \
            cap *= 2;                                                           \
            stack = (Tensor **)realloc(stack, sizeof(Tensor *) * cap);          \
        }                                                                       \
        stack[n++] = (x);                                                       \
    } while (0)

    TF_PUSH(t);
    while (n) {
        Tensor *cur = stack[--n];
        if (cur->freed) continue;

        int np = cur->nparents;
        Tensor *p0 = cur->parent[0];
        Tensor *p1 = cur->parent[1];
        Tensor *saved = cur->saved;
        OpKind op = cur->op;

        int ready = (cur->refs <= 0 && !cur->held);
        if (ready) tf_free_one(cur);

        Tensor *parents[2] = { p0, p1 };
        for (int i = 0; i < np; i++) {
            Tensor *p = parents[i];
            if (!p) continue;
            if (p->refs > 0) p->refs--;
            if (!p->freed && !p->held && p->refs <= 0) TF_PUSH(p);
        }
        if (saved && op == OP_WHERE) {
            Tensor *c = saved;
            if (c->refs > 0) c->refs--;
            if (!c->freed && !c->held && c->refs <= 0) TF_PUSH(c);
        }
    }
#undef TF_PUSH
    free(stack);
}

void tensor_free(Tensor *t) { tf_free(t); }

/* ================================================================== */
/* Accessors                                                          */
/* ================================================================== */

int     tensor_ndim(const Tensor *t) { return t->ndim; }
int64_t tensor_size(const Tensor *t, int dim) { return t->shape[dim]; }
int64_t tensor_numel(const Tensor *t) { return t->numel; }
const int64_t *tensor_shape(const Tensor *t) { return t->shape; }
float  *tensor_data(Tensor *t) { return t->data; }
float  *tensor_grad(Tensor *t) { return t->grad; }
int     tensor_requires_grad(const Tensor *t) { return t->requires_grad; }
int     tensor_opcode(const Tensor *t) { return (int)t->op; }

int tensor_same_shape(const Tensor *a, const Tensor *b) {
    if (a->ndim != b->ndim) return 0;
    for (int d = 0; d < a->ndim; d++) if (a->shape[d] != b->shape[d]) return 0;
    return 1;
}

void tensor_zero_grad(Tensor *t) {
    if (t && t->grad) memset(t->grad, 0, sizeof(float) * (size_t)t->numel);
}

static int64_t flat_index(const Tensor *t, const int64_t *idx) {
    int64_t off = 0;
    for (int d = 0; d < t->ndim; d++) off += idx[d] * t->stride[d];
    return off;
}

float tensor_get(const Tensor *t, const int64_t *idx) { return t->data[flat_index(t, idx)]; }

void tensor_set(Tensor *t, const int64_t *idx, float v) { t->data[flat_index(t, idx)] = v; }

/* ================================================================== */
/* Construction                                                       */
/* ================================================================== */

/* `held` marks a tensor as a caller-visible handle (a parameter or an input
 * constant). Freeing a loss graph reclaims the intermediates but never a held
 * tensor, which is what makes "free the graph, keep the model" work. */
Tensor *tensor_create(MiniTorchScope *s, const int64_t *shape, int ndim,
                      const float *data, int requires_grad) {
    Tensor *t = tf_new(s, shape, ndim);
    if (data) memcpy(t->data, data, sizeof(float) * (size_t)t->numel);
    t->requires_grad = requires_grad ? 1 : 0;
    t->held = 1;
    return t;
}

Tensor *tensor_zeros(MiniTorchScope *s, const int64_t *shape, int ndim) {
    return tensor_create(s, shape, ndim, NULL, 0);
}

Tensor *tensor_full(MiniTorchScope *s, const int64_t *shape, int ndim, float v) {
    Tensor *t = tf_new(s, shape, ndim);
    for (int64_t i = 0; i < t->numel; i++) t->data[i] = v;
    return t;
}

/* Same as tensor_full but a caller-visible handle. */
Tensor *tensor_constant(MiniTorchScope *s, const int64_t *shape, int ndim, float v) {
    Tensor *t = tensor_full(s, shape, ndim, v);
    t->held = 1;
    return t;
}

Tensor *tensor_clone(const Tensor *t) {
    Tensor *c = tf_new(t->scope, t->shape, t->ndim);
    memcpy(c->data, t->data, sizeof(float) * (size_t)t->numel);
    c->requires_grad = t->requires_grad;
    return c;
}

/* xorshift64* -- dependency free, deterministic, good enough for init. */
unsigned long long mt_rand_next(unsigned long long *s) {
    unsigned long long x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

float mt_rand_uniform(unsigned long long *s) {
    return (float)((mt_rand_next(s) >> 11) * (1.0 / 9007199254740992.0));
}

/* Layer constructors are convenient to call without threading an RNG through
 * every signature, so NULL means "use the process-wide stream". data.c
 * installs the real accessor; tensor.c stays standalone if it never does. */
unsigned long long *(*mt_default_rng)(void) = NULL;

static unsigned long long fallback_state = 0x9E3779B97F4A7C15ULL;
static unsigned long long *default_rng(void) {
    return mt_default_rng ? mt_default_rng() : &fallback_state;
}

Tensor *tensor_rand_uniform(MiniTorchScope *s, const int64_t *shape, int ndim,
                            float lo, float hi, unsigned long long *rng_state) {
    if (!rng_state) rng_state = default_rng();
    Tensor *t = tf_new(s, shape, ndim);
    for (int64_t i = 0; i < t->numel; i++)
        t->data[i] = lo + (hi - lo) * mt_rand_uniform(rng_state);
    t->held = 1;
    return t;
}

Tensor *tensor_rand_normal(MiniTorchScope *s, const int64_t *shape, int ndim,
                           float std, unsigned long long *rng_state) {
    if (!rng_state) rng_state = default_rng();
    Tensor *t = tf_new(s, shape, ndim);
    for (int64_t i = 0; i < t->numel; i += 2) {
        float u1 = mt_rand_uniform(rng_state), u2 = mt_rand_uniform(rng_state);
        if (u1 < 1e-7f) u1 = 1e-7f;
        float r = sqrtf(-2.0f * logf(u1));
        t->data[i] = std * r * cosf(2.0f * (float)M_PI * u2);
        if (i + 1 < t->numel) t->data[i + 1] = std * r * sinf(2.0f * (float)M_PI * u2);
    }
    t->held = 1;
    return t;
}

/* ================================================================== */
/* Raw kernels (no graph) -- needed inside backward rules             */
/* ================================================================== */

static void raw_gemm(const float *A, const float *B, float *C, int64_t M, int64_t K, int64_t N) {
    for (int64_t i = 0; i < M; i++)
        for (int64_t k = 0; k < K; k++) {
            float a = A[i * K + k];
            if (a == 0.0f) continue;
            for (int64_t j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
        }
}

/* C[M,N] += A[M,K] @ B[N,K]^T */
static void raw_gemm_tn(const float *A, const float *B, float *C, int64_t M, int64_t K, int64_t N) {
    for (int64_t i = 0; i < M; i++)
        for (int64_t j = 0; j < N; j++) {
            const float *a = A + i * K, *b = B + j * K;
            float acc = 0.0f;
            for (int64_t k = 0; k < K; k++) acc += a[k] * b[k];
            C[i * N + j] += acc;
        }
}

/* C[K,N] += A[M,K]^T @ B[M,N]
 * `M` is the CONTRACTION length and is independent of the row count of C,
 * which is K. (Passing M as the leading dimension here would walk C past its
 * end -- that bug is invisible on square-ish shapes and corrupts the heap on a
 * real convolution, where the contraction length is much larger than K.) */
static void raw_gemm_nt(const float *A, const float *B, float *C, int64_t M, int64_t K, int64_t N) {
    for (int64_t i = 0; i < M; i++)
        for (int64_t k = 0; k < K; k++) {
            float a = A[i * K + k];
            if (a == 0.0f) continue;
            float *c = C + k * N;
            const float *b = B + i * N;
            for (int64_t j = 0; j < N; j++) c[j] += a * b[j];
        }
}

/* ================================================================== */
/* Backward rules                                                     */
/* ================================================================== */

/* Generic rule: gout[i] flows to map_to_parent(i) with an optional scale. */
static void bwd_generic(Tensor *n, int which, float *g, const float *gout, float scale) {
    Tensor *p = n->parent[which];
    if (!g) return;
    /* Fast path only when the layouts are truly identical. Note that equal
     * element counts are NOT enough: a bias of shape [1,1,1,C] broadcast over
     * [N,C,H,W] has the same numel as the output in the H=W=1 case but must
     * still be summed over, so falling into an element-wise copy there would
     * overrun the bias gradient buffer. */
    int identical = (n->ndim == p->ndim);
    if (identical)
        for (int d = 0; d < n->ndim; d++)
            if (n->shape[d] != p->shape[d]) { identical = 0; break; }
    if (identical) {
        for (int64_t i = 0; i < n->numel; i++) g[i] += gout[i] * scale;
        return;
    }
    /* Generic case: walk the OUTPUT and scatter each element where it belongs.
     * Correct for broadcasting and for reductions over a single axis. */
    for (int64_t i = 0; i < n->numel; i++) {
        int64_t dst = map_to_parent(i, n, p);
        g[dst] += gout[i] * scale;
    }
}

/* Full reduction (sum to a scalar): the output gradient is a single number
 * that must be broadcast to EVERY input element. Walking the output (as the
 * generic rule does) would only touch one element, so this needs its own
 * rule that iterates the input. */
static void bwd_full_reduce(Tensor *n, int which, float *g, const float *gout) {
    Tensor *p = n->parent[which];
    if (!g) return;
    for (int64_t i = 0; i < p->numel; i++) g[i] += gout[0];
}

static void bwd_id(Tensor *n, int which, float *g, const float *gout) {
    bwd_generic(n, which, g, gout, 1.0f);
}

/* Reshape does not move data, so gradient element i of the output is gradient
 * element i of the input.
 *
 * This deliberately does NOT use the generic scatter. A node produced by a
 * rank-CHANGING reshape only fills shape[0..ndim-1]; the trailing shape[]
 * slots hold stale values from the allocation. The generic rule would try to
 * map through those non-existent axes (they look like broadcast axes) and
 * scatter the gradient to the wrong place -- and can index past the parent. */
static void bwd_reshape(Tensor *n, int which, float *g, const float *gout) {
    Tensor *p = n->parent[which];
    if (!g) return;
    int64_t count = n->numel < p->numel ? n->numel : p->numel;
    for (int64_t i = 0; i < count; i++) g[i] += gout[i];
}

static void bwd_neg(Tensor *n, int which, float *g, const float *gout) {
    (void)n; (void)which;
    for (int64_t i = 0; i < n->numel; i++) g[i] -= gout[i];
}

static void bwd_relu(Tensor *n, int which, float *g, const float *gout) {
    (void)which;
    for (int64_t i = 0; i < n->numel; i++) g[i] += gout[i] * (n->data[i] > 0.0f ? 1.0f : 0.0f);
}
static void bwd_sigmoid(Tensor *n, int which, float *g, const float *gout) {
    (void)which;
    for (int64_t i = 0; i < n->numel; i++) { float y = n->data[i]; g[i] += gout[i] * y * (1.0f - y); }
}
static void bwd_tanh(Tensor *n, int which, float *g, const float *gout) {
    (void)which;
    for (int64_t i = 0; i < n->numel; i++) { float y = n->data[i]; g[i] += gout[i] * (1.0f - y * y); }
}
static void bwd_gelu(Tensor *n, int which, float *g, const float *gout) {
    (void)which;
    const float c = 0.7978845608f, k = 0.044715f;
    const float *x = n->parent[0]->data;
    for (int64_t i = 0; i < n->numel; i++) {
        float v = x[i], v2 = v * v;
        float th = tanhf(c * (v + k * v2 * v));
        float d = 0.5f * (1.0f + th) + 0.5f * v * (1.0f - th * th) * c * (1.0f + 3.0f * k * v2);
        g[i] += gout[i] * d;
    }
}
static void bwd_exp(Tensor *n, int which, float *g, const float *gout) {
    (void)which;
    for (int64_t i = 0; i < n->numel; i++) g[i] += gout[i] * n->data[i];
}
static void bwd_log(Tensor *n, int which, float *g, const float *gout) {
    (void)which;
    const float *x = n->parent[0]->data;
    for (int64_t i = 0; i < n->numel; i++) g[i] += gout[i] / x[i];
}
static void bwd_sqrt(Tensor *n, int which, float *g, const float *gout) {
    (void)which;
    for (int64_t i = 0; i < n->numel; i++) g[i] += gout[i] * 0.5f / n->data[i];
}
static void bwd_pow(Tensor *n, int which, float *g, const float *gout) {
    (void)which;
    const float *x = n->parent[0]->data;
    float p = n->scalar;
    for (int64_t i = 0; i < n->numel; i++) g[i] += gout[i] * p * powf(x[i], p - 1.0f);
}

/* binary rules ----------------------------------------------------- */

static void bwd_add(Tensor *n, float *ga, float *gb, const float *gout) {
    bwd_generic(n, 0, ga, gout, 1.0f);
    bwd_generic(n, 1, gb, gout, 1.0f);
}
static void bwd_sub(Tensor *n, float *ga, float *gb, const float *gout) {
    bwd_generic(n, 0, ga, gout, 1.0f);
    bwd_generic(n, 1, gb, gout, -1.0f);
}
static void bwd_mul(Tensor *n, float *ga, float *gb, const float *gout) {
    Tensor *A = n->parent[0], *B = n->parent[1];
    for (int64_t i = 0; i < n->numel; i++) {
        int64_t ia = map_to_parent(i, n, A), ib = map_to_parent(i, n, B);
        if (ga) ga[ia] += gout[i] * B->data[ib];
        if (gb) gb[ib] += gout[i] * A->data[ia];
    }
}
static void bwd_div(Tensor *n, float *ga, float *gb, const float *gout) {
    Tensor *A = n->parent[0], *B = n->parent[1];
    for (int64_t i = 0; i < n->numel; i++) {
        int64_t ia = map_to_parent(i, n, A), ib = map_to_parent(i, n, B);
        if (ga) ga[ia] += gout[i] / B->data[ib];
        if (gb) gb[ib] -= gout[i] * A->data[ia] / (B->data[ib] * B->data[ib]);
    }
}
static void bwd_maximum(Tensor *n, float *ga, float *gb, const float *gout) {
    Tensor *A = n->parent[0], *B = n->parent[1];
    for (int64_t i = 0; i < n->numel; i++) {
        int64_t ia = map_to_parent(i, n, A), ib = map_to_parent(i, n, B);
        float a = A->data[ia], b = B->data[ib];
        if (a > b) { if (ga) ga[ia] += gout[i]; }
        else if (b > a) { if (gb) gb[ib] += gout[i]; }
        else { if (ga) ga[ia] += 0.5f * gout[i]; if (gb) gb[ib] += 0.5f * gout[i]; }
    }
}
static void bwd_where(Tensor *n, float *ga, float *gb, const float *gout) {
    const Tensor *cond = n->saved;
    for (int64_t i = 0; i < n->numel; i++) {
        int64_t ic = map_to_parent(i, n, cond);
        int64_t ia = map_to_parent(i, n, n->parent[0]);
        int64_t ib = map_to_parent(i, n, n->parent[1]);
        if (cond->data[ic] != 0.0f) { if (ga) ga[ia] += gout[i]; }
        else { if (gb) gb[ib] += gout[i]; }
    }
}

/* reduction / reshaping rules -------------------------------------- */

static void bwd_sum_dim(Tensor *n, int which, float *g, const float *gout) {
    Tensor *p = n->parent[which];
    int dim = (int)n->scalar;
    int64_t rest = 1;
    for (int d = dim + 1; d < p->ndim; d++) rest *= p->shape[d];
    int64_t outer = p->shape[dim] * rest;
    for (int64_t i = 0; i < p->numel; i++) {
        int64_t o = (i / outer) * rest + (i % rest);
        g[i] += gout[o];
    }
}
static void bwd_mean_dim(Tensor *n, int which, float *g, const float *gout) {
    Tensor *p = n->parent[which];
    int dim = (int)n->scalar;
    int64_t rest = 1;
    for (int d = dim + 1; d < p->ndim; d++) rest *= p->shape[d];
    int64_t outer = p->shape[dim] * rest;
    float inv = 1.0f / (float)p->shape[dim];
    for (int64_t i = 0; i < p->numel; i++) {
        int64_t o = (i / outer) * rest + (i % rest);
        g[i] += gout[o] * inv;
    }
}
static void bwd_transpose(Tensor *n, int which, float *g, const float *gout) {
    Tensor *p = n->parent[which];
    int64_t rows = p->shape[0], cols = p->shape[1];
    for (int64_t i = 0; i < rows; i++)
        for (int64_t j = 0; j < cols; j++) g[i * cols + j] += gout[j * rows + i];
}

/* matmul / conv / pool / embedding / dropout ------------------------ */

static void bwd_matmul(Tensor *n, float *ga, float *gb, const float *gout) {
    Tensor *A = n->parent[0], *B = n->parent[1];   /* [M,K] x [K,N] */
    int64_t M = A->shape[0], K = A->shape[1], N = B->shape[1];
    if (ga) raw_gemm_tn(gout, B->data, ga, M, N, K);   /* dA = gout @ B^T  [M,K] */
    if (gb) raw_gemm_nt(A->data, gout, gb, M, K, N);   /* dB = A^T @ gout  [K,N] */
}
static void bwd_matmul_tn(Tensor *n, float *ga, float *gb, const float *gout) {
    Tensor *A = n->parent[0], *B = n->parent[1];   /* C[M,N] = A[M,K] @ B[N,K]^T */
    int64_t M = A->shape[0], K = A->shape[1], N = B->shape[0];
    if (ga) raw_gemm(gout, B->data, ga, M, N, K);      /* dA = gout @ B      [M,K] */
    /* dB = gout^T @ A is [N,K]. raw_gemm_nt takes A as [M,K] and B as [M,N],
     * so the roles swap: gout is the [M,N] operand and A supplies the columns. */
    if (gb) raw_gemm_nt(gout, A->data, gb, M, N, K);
}

typedef struct {
    int64_t N, C, H, W, K, stride, pad, dil, Ho, Wo;
    int64_t *col_of_input; /* per (n,oh,ow,c,kh,kw) source index, -1 = padding */
} Im2ColCtx;

static void bwd_im2col(Tensor *n, int which, float *g, const float *gout) {
    (void)which;
    Im2ColCtx *c = (Im2ColCtx *)n->ctx;
    int64_t rows = c->N * c->Ho * c->Wo, cols = c->C * c->K * c->K;
    for (int64_t r = 0; r < rows; r++)
        for (int64_t j = 0; j < cols; j++) {
            int64_t dst = c->col_of_input[r * cols + j];
            if (dst >= 0) g[dst] += gout[r * cols + j];
        }
}

typedef struct { int64_t *arg; } PoolCtx;

static void bwd_maxpool(Tensor *n, int which, float *g, const float *gout) {
    (void)which;
    PoolCtx *c = (PoolCtx *)n->ctx;
    for (int64_t i = 0; i < n->numel; i++) {
        int64_t src = c->arg[i];
        if (src >= 0) g[src] += gout[i];
    }
}

/* ctx holds a plain int64 buffer of the looked-up rows. It must NOT be a
 * registered Tensor: tf_free_one already frees ctx, and a registered tensor
 * would also be freed by its scope. */
static void bwd_embedding(Tensor *n, int which, float *g, const float *gout) {
    (void)which;
    const int64_t *idx = (const int64_t *)n->ctx;
    Tensor *w = n->parent[0];
    int64_t V = w->shape[0], D = w->shape[1];
    int64_t rows = n->numel / D;
    for (int64_t r = 0; r < rows; r++) {
        int64_t tok = idx[r];
        if (tok < 0 || tok >= V) continue;
        for (int64_t d = 0; d < D; d++) g[tok * D + d] += gout[r * D + d];
    }
}

static void bwd_dropout(Tensor *n, int which, float *g, const float *gout) {
    (void)which;
    const float *mask = (const float *)n->ctx;
    for (int64_t i = 0; i < n->numel; i++) g[i] += gout[i] * mask[i];
}

static void bwd_add_bias(Tensor *n, float *gx, float *gb, const float *gout) {
    Tensor *x = n->parent[0], *b = n->parent[1];
    int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    if (gx)
        for (int64_t i = 0; i < x->numel; i++) gx[i] += gout[i];
    if (gb)
        for (int64_t nn = 0; nn < N; nn++)
            for (int64_t c = 0; c < C; c++) {
                float acc = 0.0f;
                int64_t base = ((nn * C + c) * H) * W;
                for (int64_t k = 0; k < H * W; k++) acc += gout[base + k];
                gb[c] += acc;
            }
    (void)b;
}

/* softmax / cross entropy ------------------------------------------ */

static void bwd_softmax(Tensor *n, int which, float *g, const float *gout) {
    (void)which;
    int64_t K = n->shape[n->ndim - 1];
    int64_t rows = n->numel / K;
    for (int64_t r = 0; r < rows; r++) {
        const float *y = n->data + r * K;
        const float *gy = gout + r * K;
        float dot = 0.0f;
        for (int64_t k = 0; k < K; k++) dot += gy[k] * y[k];
        float *gx = g + r * K;
        for (int64_t k = 0; k < K; k++) gx[k] += y[k] * (gy[k] - dot);
    }
}

/* n->ctx = one-hot targets [N,C]; n->data = log-softmax probs */
static void bwd_cross_entropy(Tensor *n, int which, float *g, const float *gout) {
    (void)which;
    Tensor *logits = n->parent[0];
    const Tensor *tgt = (const Tensor *)n->saved;
    int64_t N = logits->shape[0], C = logits->shape[1];
    float scale = (n->scalar > 0.5f) ? (1.0f / (float)N) : 1.0f; /* 1 = mean, 0 = sum */
    float g0 = gout[0] * scale;
    const float *logp = (const float *)n->ctx;
    for (int64_t i = 0; i < N; i++)
        for (int64_t c = 0; c < C; c++) {
            float prob = expf(logp[i * C + c]);
            g[i * C + c] += g0 * (prob - tgt->data[i * C + c]);
        }
}

const char *tensor_op_name(const Tensor *t) { return g_ops[t->op].name; }

/* ================================================================== */
/* Elementwise ops                                                    */
/* ================================================================== */

typedef float (*UnaryFn)(float, void *);

static float u_relu(float x, void *u)    { (void)u; return x > 0.0f ? x : 0.0f; }
static float u_sigmoid(float x, void *u) { (void)u; return 1.0f / (1.0f + expf(-x)); }
static float u_tanh(float x, void *u)    { (void)u; return tanhf(x); }
static float u_gelu(float x, void *u) {
    (void)u;
    const float c = 0.7978845608f, k = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(c * (x + k * x * x * x)));
}
static float u_exp(float x, void *u)  { (void)u; return expf(x); }
static float u_sqrt(float x, void *u) { (void)u; return sqrtf(x); }
static float u_pow(float x, void *u)  { return powf(x, *(float *)u); }
static float u_neg(float x, void *u)  { (void)u; return -x; }

static Tensor *tf_unary(MiniTorchScope *s, Tensor *a, OpKind op, UnaryFn fn, void *ud) {
    Tensor *out = tf_result(s, a->shape, a->ndim, op);
    for (int64_t i = 0; i < a->numel; i++) out->data[i] = fn(a->data[i], ud);
    tf_attach(out, a, NULL);
    if (op == OP_RELU || op == OP_GELU) out->is_leaf = 0;
    return out;
}

typedef float (*BinaryFn)(float, float);

static float bf_add(float a, float b) { return a + b; }
static float bf_sub(float a, float b) { return a - b; }
static float bf_mul(float a, float b) { return a * b; }
static float bf_div(float a, float b) { return a / b; }
static float bf_max(float a, float b) { return a > b ? a : b; }

static int64_t *bcast_shape(const Tensor *a, const Tensor *b, int *ndim_out) {
    int nd = a->ndim > b->ndim ? a->ndim : b->ndim;
    int64_t *sh = (int64_t *)malloc(sizeof(int64_t) * (size_t)(nd > 0 ? nd : 1));
    for (int d = 0; d < nd; d++) {
        /* Broadcasting aligns axes from the RIGHT: the trailing axis of the
         * lower-rank operand lines up with the trailing axis of the higher. */
        int da = d - (nd - a->ndim);
        int db = d - (nd - b->ndim);
        int64_t sa = da >= 0 ? a->shape[da] : 1;
        int64_t sb = db >= 0 ? b->shape[db] : 1;
        if (sa != sb && sa != 1 && sb != 1) {
            fprintf(stderr, "[tensor] cannot broadcast [");
            for (int k = 0; k < a->ndim; k++)
                fprintf(stderr, "%lld%s", (long long)a->shape[k], k + 1 < a->ndim ? "," : "");
            fprintf(stderr, "] with [");
            for (int k = 0; k < b->ndim; k++)
                fprintf(stderr, "%lld%s", (long long)b->shape[k], k + 1 < b->ndim ? "," : "");
            fprintf(stderr, "] (axis %d: %lld vs %lld)\n", d, (long long)sa, (long long)sb);
            free(sh);
            *ndim_out = -1;
            return NULL;
        }
        sh[d] = sa > sb ? sa : sb;
    }
    *ndim_out = nd;
    return sh;
}

static Tensor *tf_binary(MiniTorchScope *s, Tensor *a, Tensor *b, OpKind op, BinaryFn fn) {
    int nd = 0;
    int64_t *sh = bcast_shape(a, b, &nd);
    if (!sh) return NULL;
    Tensor *out = tf_result(s, sh, nd, op);
    free(sh);
    contiguous_strides(out->shape, out->ndim, out->stride);
    for (int64_t i = 0; i < out->numel; i++) {
        int64_t ia = map_to_parent(i, out, a);
        int64_t ib = map_to_parent(i, out, b);
        out->data[i] = fn(a->data[ia], b->data[ib]);
    }
    tf_attach(out, a, b);
    return out;
}

/* ================================================================== */
/* Public elementwise / shape API                                     */
/* ================================================================== */

Tensor *tensor_add(Tensor *a, Tensor *b) { return tf_binary(a->scope, a, b, OP_ADD, bf_add); }
Tensor *tensor_sub(Tensor *a, Tensor *b) { return tf_binary(a->scope, a, b, OP_SUB, bf_sub); }
Tensor *tensor_mul(Tensor *a, Tensor *b) { return tf_binary(a->scope, a, b, OP_MUL, bf_mul); }
Tensor *tensor_div(Tensor *a, Tensor *b) { return tf_binary(a->scope, a, b, OP_DIV, bf_div); }
Tensor *tensor_maximum(Tensor *a, Tensor *b) { return tf_binary(a->scope, a, b, OP_MAXIMUM, bf_max); }

Tensor *tensor_neg(Tensor *a) { return tf_unary(a->scope, a, OP_NEG, u_neg, NULL); }
Tensor *tensor_relu(Tensor *a) { return tf_unary(a->scope, a, OP_RELU, u_relu, NULL); }
Tensor *tensor_sigmoid(Tensor *a) { return tf_unary(a->scope, a, OP_SIGMOID, u_sigmoid, NULL); }
Tensor *tensor_tanh(Tensor *a) { return tf_unary(a->scope, a, OP_TANH, u_tanh, NULL); }
Tensor *tensor_gelu(Tensor *a) { return tf_unary(a->scope, a, OP_GELU, u_gelu, NULL); }
Tensor *tensor_exp(Tensor *a) { return tf_unary(a->scope, a, OP_EXP, u_exp, NULL); }
Tensor *tensor_sqrt(Tensor *a) { return tf_unary(a->scope, a, OP_SQRT, u_sqrt, NULL); }

Tensor *tensor_pow(Tensor *a, float p) {
    Tensor *out = tf_unary(a->scope, a, OP_POW, u_pow, &p);
    out->scalar = p;
    return out;
}

Tensor *tensor_log(Tensor *a) {
    Tensor *out = tf_result(a->scope, a->shape, a->ndim, OP_LOG);
    for (int64_t i = 0; i < a->numel; i++) out->data[i] = logf(a->data[i]);
    tf_attach(out, a, NULL);
    return out;
}

Tensor *tensor_add_scalar(Tensor *a, float s) {
    Tensor *sc = tensor_full(a->scope, &(int64_t){1}, 1, s);
    return tf_binary(a->scope, a, sc, OP_ADD, bf_add);
}

Tensor *tensor_mul_scalar(Tensor *a, float s) {
    Tensor *sc = tensor_full(a->scope, &(int64_t){1}, 1, s);
    return tf_binary(a->scope, a, sc, OP_MUL, bf_mul);
}

Tensor *tensor_where(const Tensor *cond, Tensor *a, Tensor *b) {
    int nd = 0;
    int64_t *sh = bcast_shape(a, b, &nd);
    if (!sh) return NULL;
    Tensor *out = tf_result(a->scope, sh, nd, OP_WHERE);
    free(sh);
    for (int64_t i = 0; i < out->numel; i++) {
        int64_t ic = map_to_parent(i, out, cond);
        int64_t ia = map_to_parent(i, out, a);
        int64_t ib = map_to_parent(i, out, b);
        out->data[i] = cond->data[ic] != 0.0f ? a->data[ia] : b->data[ib];
    }
    tf_attach(out, a, b);
    tf_save(out, (Tensor *)cond);
    return out;
}

Tensor *tensor_sum(Tensor *a) {
    /* Kept as shape [1] rather than a rank-0 tensor: every op in the engine
     * works on at least one axis, and this avoids special cases everywhere. */
    int64_t sh[1] = { 1 };
    Tensor *out = tf_result(a->scope, sh, 1, OP_SUM);
    float acc = 0.0f;
    for (int64_t i = 0; i < a->numel; i++) acc += a->data[i];
    out->data[0] = acc;
    tf_attach(out, a, NULL);
    return out;
}

Tensor *tensor_sum_dim(Tensor *a, int dim, int keepdim) {
    if (dim < 0 || dim >= a->ndim) { fprintf(stderr, "[tensor] sum_dim: bad dim\n"); return NULL; }
    int rnd = keepdim ? a->ndim : (a->ndim > 1 ? a->ndim - 1 : 1);
    int64_t sh[TF_MAXDIM];
    if (keepdim) {
        for (int d = 0; d < a->ndim; d++) sh[d] = a->shape[d];
        sh[dim] = 1;
    } else {
        int k = 0;
        for (int d = 0; d < a->ndim; d++) if (d != dim) sh[k++] = a->shape[d];
        if (k == 0) sh[k++] = 1;
    }
    Tensor *out = tf_result(a->scope, sh, rnd, OP_SUM_DIM);
    out->scalar = (float)dim;
    int64_t rest = 1;
    for (int d = dim + 1; d < a->ndim; d++) rest *= a->shape[d];
    int64_t outer = a->shape[dim] * rest;
    for (int64_t i = 0; i < a->numel; i++) {
        int64_t o = (i / outer) * rest + (i % rest);
        out->data[o] += a->data[i];
    }
    tf_attach(out, a, NULL);
    return out;
}

Tensor *tensor_mean_dim(Tensor *a, int dim, int keepdim) {
    if (dim < 0 || dim >= a->ndim) { fprintf(stderr, "[tensor] mean_dim: bad dim\n"); return NULL; }
    Tensor *s = tensor_sum_dim(a, dim, keepdim);
    Tensor *out = tf_result(a->scope, s->shape, s->ndim, OP_MEAN_DIM);
    out->scalar = (float)dim;
    float inv = 1.0f / (float)a->shape[dim];
    for (int64_t i = 0; i < s->numel; i++) out->data[i] = s->data[i] * inv;
    tf_attach(out, a, NULL);
    return out;
}

Tensor *tensor_reshape(Tensor *a, const int64_t *shape, int ndim) {
    int64_t want = numel_of(shape, ndim), have = a->numel;
    if (want != have) {
        fprintf(stderr, "[tensor] reshape numel mismatch (%lld vs %lld)\n", (long long)want,
                (long long)have);
        return NULL;
    }
    Tensor *out = tf_result(a->scope, shape, ndim, OP_RESHAPE);
    memcpy(out->data, a->data, sizeof(float) * (size_t)a->numel);
    tf_attach(out, a, NULL);
    return out;
}

Tensor *tensor_transpose2d(Tensor *a) {
    if (a->ndim != 2) { fprintf(stderr, "[tensor] transpose2d wants 2-D\n"); return NULL; }
    int64_t sh[2] = { a->shape[1], a->shape[0] };
    Tensor *out = tf_result(a->scope, sh, 2, OP_TRANSPOSE);
    for (int64_t i = 0; i < sh[0]; i++)
        for (int64_t j = 0; j < sh[1]; j++)
            out->data[i * sh[1] + j] = a->data[j * a->shape[1] + i];
    tf_attach(out, a, NULL);
    return out;
}

Tensor *tensor_broadcast_to(Tensor *a, const int64_t *shape, int ndim) {
    Tensor *out = tf_result(a->scope, shape, ndim, OP_BROADCAST);
    for (int64_t i = 0; i < out->numel; i++)
        out->data[i] = a->data[map_to_parent(i, out, a)];
    tf_attach(out, a, NULL);
    return out;
}

Tensor *tensor_matmul(Tensor *a, Tensor *b) {
    if (a->ndim != 2 || b->ndim != 2) { fprintf(stderr, "[tensor] matmul wants 2-D\n"); return NULL; }
    if (a->shape[1] != b->shape[0]) {
        fprintf(stderr, "[tensor] matmul mismatch [%lld,%lld]x[%lld,%lld]\n", (long long)a->shape[0],
                (long long)a->shape[1], (long long)b->shape[0], (long long)b->shape[1]);
        return NULL;
    }
    int64_t sh[2] = { a->shape[0], b->shape[1] };
    Tensor *out = tf_result(a->scope, sh, 2, OP_MATMUL);
    raw_gemm(a->data, b->data, out->data, sh[0], a->shape[1], sh[1]);
    tf_attach(out, a, b);
    return out;
}

Tensor *tensor_matmul_tn(Tensor *a, Tensor *b) {
    if (a->ndim != 2 || b->ndim != 2 || a->shape[1] != b->shape[1]) {
        fprintf(stderr, "[tensor] matmul_tn mismatch\n");
        return NULL;
    }
    int64_t sh[2] = { a->shape[0], b->shape[0] };
    Tensor *out = tf_result(a->scope, sh, 2, OP_MATMUL_TN);
    raw_gemm_tn(a->data, b->data, out->data, sh[0], a->shape[1], sh[1]);
    tf_attach(out, a, b);
    return out;
}

/* ================================================================== */
/* im2col / conv2d / maxpool                                          */
/* ================================================================== */

Tensor *tensor_im2col(Tensor *x, int K, int stride, int pad, int dil) {
    if (x->ndim != 4) { fprintf(stderr, "[tensor] im2col wants NCHW\n"); return NULL; }
    Im2ColCtx *c = (Im2ColCtx *)calloc(1, sizeof(Im2ColCtx));
    c->N = x->shape[0]; c->C = x->shape[1]; c->H = x->shape[2]; c->W = x->shape[3];
    c->K = K; c->stride = stride; c->pad = pad; c->dil = dil;
    c->Ho = (c->H + 2 * pad - dil * (K - 1) - 1) / stride + 1;
    c->Wo = (c->W + 2 * pad - dil * (K - 1) - 1) / stride + 1;
    if (c->Ho <= 0 || c->Wo <= 0) {
        fprintf(stderr, "[tensor] im2col: kernel larger than padded input\n");
        free(c);
        return NULL;
    }
    int64_t rows = c->N * c->Ho * c->Wo, cols = c->C * K * K;
    c->col_of_input = (int64_t *)malloc(sizeof(int64_t) * (size_t)(rows * cols));
    int64_t p = 0;
    for (int64_t n = 0; n < c->N; n++)
        for (int64_t oh = 0; oh < c->Ho; oh++)
            for (int64_t ow = 0; ow < c->Wo; ow++)
                for (int64_t ch = 0; ch < c->C; ch++)
                    for (int64_t kh = 0; kh < K; kh++)
                        for (int64_t kw = 0; kw < K; kw++) {
                            int64_t ih = oh * stride - pad + kh * dil;
                            int64_t iw = ow * stride - pad + kw * dil;
                            int64_t idx = -1;
                            if (ih >= 0 && ih < c->H && iw >= 0 && iw < c->W)
                                idx = ((n * c->C + ch) * c->H + ih) * c->W + iw;
                            c->col_of_input[p++] = idx;
                        }
    int64_t sh[2] = { rows, cols };
    Tensor *out = tf_result(x->scope, sh, 2, OP_IM2COL);
    for (int64_t i = 0; i < rows * cols; i++) {
        int64_t src = c->col_of_input[i];
        out->data[i] = src < 0 ? 0.0f : x->data[src];
    }
    tf_attach(out, x, NULL);
    out->ctx = c;
    return out;
}

/* permute [N,Ho,Wo,O] -> [N,O,Ho,Wo] */
static Tensor *permute_to_nchw(Tensor *src, int64_t N, int64_t O, int64_t Ho, int64_t Wo) {
    int64_t sh[4] = { N, O, Ho, Wo };
    Tensor *out = tf_result(src->scope, sh, 4, OP_PERMUTE);
    for (int64_t n = 0; n < N; n++)
        for (int64_t o = 0; o < O; o++)
            for (int64_t h = 0; h < Ho; h++)
                for (int64_t w = 0; w < Wo; w++)
                    out->data[((n * O + o) * Ho + h) * Wo + w] =
                        src->data[((n * Ho + h) * Wo + w) * O + o];
    tf_attach(out, src, NULL);
    return out;
}

static void bwd_permute(Tensor *n, int which, float *g, const float *gout) {
    (void)which;
    Tensor *p = n->parent[0];
    int64_t N = p->shape[0], Ho = p->shape[1], Wo = p->shape[2], O = p->shape[3];
    for (int64_t n_ = 0; n_ < N; n_++)
        for (int64_t o = 0; o < O; o++)
            for (int64_t h = 0; h < Ho; h++)
                for (int64_t w = 0; w < Wo; w++)
                    g[((n_ * Ho + h) * Wo + w) * O + o] =
                        gout[((n_ * O + o) * Ho + h) * Wo + w];
}

Tensor *tensor_conv2d(Tensor *x, Tensor *w, Tensor *b, int stride, int padding, int dilation) {
    if (x->ndim != 4 || w->ndim != 4) {
        fprintf(stderr, "[tensor] conv2d expects NCHW input and [O,C,Kh,Kw] weight\n");
        return NULL;
    }
    int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    int64_t O = w->shape[0], K = w->shape[2];
    int64_t Ho = (H + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    int64_t Wo = (W + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    Tensor *col = tensor_im2col(x, (int)K, stride, padding, dilation);
    if (!col) return NULL;
    int64_t wsh[2] = { O, C * K * K };
    Tensor *wf = tensor_reshape(w, wsh, 2);
    Tensor *wT = tensor_transpose2d(wf);
    Tensor *mm = tensor_matmul(col, wT);              /* [N*Ho*Wo, O] */
    int64_t osh[4] = { N, Ho, Wo, O };
    Tensor *r4 = tensor_reshape(mm, osh, 4);
    Tensor *add = r4;
    if (b && b->numel == O) {
        int64_t bsh[4] = { 1, 1, 1, O };
        Tensor *bb = tensor_reshape(b, bsh, 4);
        add = tensor_add(r4, bb);
    }
    Tensor *nchw = permute_to_nchw(add, N, O, Ho, Wo);
    return nchw;
}

Tensor *tensor_maxpool2d(Tensor *x, int k, int stride) {
    if (x->ndim != 4) { fprintf(stderr, "[tensor] maxpool wants NCHW\n"); return NULL; }
    int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    if (stride <= 0) stride = k;
    int64_t Ho = (H - k) / stride + 1, Wo = (W - k) / stride + 1;
    if (Ho <= 0 || Wo <= 0) { fprintf(stderr, "[tensor] maxpool: kernel too large\n"); return NULL; }
    int64_t sh[4] = { N, C, Ho, Wo };
    Tensor *out = tf_result(x->scope, sh, 4, OP_MAXPOOL);
    PoolCtx *ctx = (PoolCtx *)calloc(1, sizeof(PoolCtx));
    ctx->arg = (int64_t *)malloc(sizeof(int64_t) * (size_t)out->numel);
    for (int64_t n = 0; n < N; n++)
        for (int64_t c = 0; c < C; c++)
            for (int64_t oh = 0; oh < Ho; oh++)
                for (int64_t ow = 0; ow < Wo; ow++) {
                    float best = -1e30f;
                    int64_t arg = -1;
                    for (int kh = 0; kh < k; kh++)
                        for (int kw = 0; kw < k; kw++) {
                            int64_t ih = oh * stride + kh, iw = ow * stride + kw;
                            int64_t src = ((n * C + c) * H + ih) * W + iw;
                            if (x->data[src] > best) { best = x->data[src]; arg = src; }
                        }
                    int64_t dst = ((n * C + c) * Ho + oh) * Wo + ow;
                    out->data[dst] = best;
                    ctx->arg[dst] = arg;
                }
    tf_attach(out, x, NULL);
    out->ctx = ctx;
    return out;
}

Tensor *tensor_embedding(Tensor *weight, const int64_t *idx, int64_t n_idx) {
    if (weight->ndim != 2) { fprintf(stderr, "[tensor] embedding weight must be [V,D]\n"); return NULL; }
    int64_t V = weight->shape[0], D = weight->shape[1];
    int64_t sh[2] = { n_idx, D };
    Tensor *out = tf_result(weight->scope, sh, 2, OP_EMBEDDING);
    int64_t *cache = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n_idx > 0 ? n_idx : 1));
    for (int64_t i = 0; i < n_idx; i++) {
        int64_t tok = idx[i];
        if (tok < 0 || tok >= V) tok = 0;
        cache[i] = tok;
        memcpy(out->data + i * D, weight->data + tok * D, sizeof(float) * (size_t)D);
    }
    tf_attach(out, weight, NULL);
    out->ctx = cache;
    return out;
}

Tensor *tensor_dropout(Tensor *x, float p, int training, unsigned long long *rng) {
    Tensor *out = tf_result(x->scope, x->shape, x->ndim, OP_DROPOUT);
    float *mask = (float *)calloc((size_t)(x->numel > 0 ? x->numel : 1), sizeof(float));
    float keep = 1.0f - p;
    if (keep <= 0.0f) keep = 1e-8f;
    for (int64_t i = 0; i < x->numel; i++) {
        if (!training) { out->data[i] = x->data[i]; mask[i] = 1.0f; continue; }
        float u = mt_rand_uniform(rng);
        float m = u < keep ? 1.0f / keep : 0.0f;
        mask[i] = m;
        out->data[i] = x->data[i] * m;
    }
    tf_attach(out, x, NULL);
    out->ctx = mask;
    return out;
}

Tensor *tensor_add_bias(Tensor *x, Tensor *b) {
    if (x->ndim != 4 || b->numel != x->shape[1]) {
        fprintf(stderr, "[tensor] add_bias expects x[N,C,H,W] and b[C]\n");
        return NULL;
    }
    Tensor *out = tf_result(x->scope, x->shape, x->ndim, OP_ADD_BIAS);
    int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    for (int64_t n = 0; n < N; n++)
        for (int64_t c = 0; c < C; c++)
            for (int64_t i = 0; i < H * W; i++)
                out->data[((n * C + c) * H * W) + i] = x->data[((n * C + c) * H * W) + i] + b->data[c];
    tf_attach(out, x, b);
    return out;
}

Tensor *tensor_softmax(Tensor *x) {
    if (x->ndim < 1) { fprintf(stderr, "[tensor] softmax needs >=1 dim\n"); return NULL; }
    Tensor *out = tf_result(x->scope, x->shape, x->ndim, OP_SOFTMAX);
    int64_t K = x->shape[x->ndim - 1];
    int64_t rows = x->numel / (K > 0 ? K : 1);
    for (int64_t r = 0; r < rows; r++) {
        const float *src = x->data + r * K;
        float *dst = out->data + r * K;
        float mx = src[0];
        for (int64_t k = 1; k < K; k++) if (src[k] > mx) mx = src[k];
        float sum = 0.0f;
        for (int64_t k = 0; k < K; k++) { dst[k] = expf(src[k] - mx); sum += dst[k]; }
        float inv = 1.0f / sum;
        for (int64_t k = 0; k < K; k++) dst[k] *= inv;
    }
    tf_attach(out, x, NULL);
    return out;
}

/* Stable log-softmax followed by NLL against one-hot targets. `reduction`
 * is 1 for mean, 0 for sum -- stored in scalar for the backward rule. */
Tensor *tensor_cross_entropy(Tensor *logits, Tensor *targets, int reduction) {
    if (logits->ndim != 2 || targets->ndim != 2 ||
        logits->shape[0] != targets->shape[0] || logits->shape[1] != targets->shape[1]) {
        fprintf(stderr, "[tensor] cross_entropy expects [N,C] logits and [N,C] one-hot targets\n");
        return NULL;
    }
    int64_t N = logits->shape[0], C = logits->shape[1];
    Tensor *out = tf_result(logits->scope, &(int64_t){1}, 1, OP_CROSS_ENTROPY);
    out->scalar = reduction ? 1.0f : 0.0f;

    /* store log-probabilities (shifted) in the output's data buffer via a
     * companion allocation kept in ctx so backward can rebuild softmax */
    float *logp = (float *)malloc(sizeof(float) * (size_t)(N * C));
    float total = 0.0f;
    for (int64_t i = 0; i < N; i++) {
        const float *z = logits->data + i * C;
        float mx = z[0];
        for (int64_t c = 1; c < C; c++) if (z[c] > mx) mx = z[c];
        float sum = 0.0f;
        for (int64_t c = 0; c < C; c++) { float e = expf(z[c] - mx); logp[i * C + c] = e; sum += e; }
        float logsum = logf(sum) + mx;
        float acc = 0.0f;
        for (int64_t c = 0; c < C; c++) {
            logp[i * C + c] = logf(logp[i * C + c]) + mx - logsum; /* log-softmax */
            acc += targets->data[i * C + c] * logp[i * C + c];
        }
        total += -acc;
    }
    out->ctx = logp;
    out->data[0] = reduction ? total / (float)N : total;
    tf_attach(out, logits, NULL);
    tf_save(out, targets);
    return out;
}

/* ================================================================== */
/* Reverse-mode autodiff driver                                       */
/* ================================================================== */

typedef struct { Tensor **items; size_t n, cap; } VisitList;

static void vl_push(VisitList *v, Tensor *t) {
    if (v->n == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 64;
        v->items = (Tensor **)realloc(v->items, sizeof(Tensor *) * v->cap);
    }
    v->items[v->n++] = t;
}

/* Topological sort (reverse post-order DFS); ensures consumers appear before producers in DAGs. */
static void build_postorder(Tensor *root, VisitList *out) {
    typedef struct { Tensor *t; int next_child; } Frame;
    size_t fcap = 64, fn = 0;
    Frame *fstack = (Frame *)malloc(sizeof(Frame) * fcap);

#define FRAME_PUSH(node) do { \
    if (fn >= fcap) { \
        fcap *= 2; \
        fstack = (Frame *)realloc(fstack, sizeof(Frame) * fcap); \
    } \
    fstack[fn++] = (Frame){ (node), 0 }; \
} while (0)

    FRAME_PUSH(root);
    root->mark = 1;

    while (fn) {
        Frame *top = &fstack[fn - 1];
        Tensor *cur = top->t;

        int found = 0;
        while (top->next_child < cur->nparents) {
            Tensor *p = cur->parent[top->next_child++];
            if (p && !p->freed && !p->mark) {
                p->mark = 1;
                FRAME_PUSH(p);
                found = 1;
                break;
            }
        }
        if (!found) {
            vl_push(out, cur);
            fn--;
        }
    }

    free(fstack);
#undef FRAME_PUSH

    /* Reverse post-order: root (output) first, leaves last, consumers before producers */
    if (out->n > 1) {
        for (size_t i = 0, j = out->n - 1; i < j; i++, j--) {
            Tensor *tmp = out->items[i];
            out->items[i] = out->items[j];
            out->items[j] = tmp;
        }
    }
}

void tensor_backward(Tensor *out, const float *seed) {
    if (!out || out->freed) return;
    if (!out->requires_grad) {
        fprintf(stderr, "[autograd] backward() on a tensor that does not require grad\n");
        return;
    }
    VisitList post = { NULL, 0, 0 };
    build_postorder(out, &post);
    for (size_t i = 0; i < post.n; i++) post.items[i]->mark = 0;

    grad_ensure(out);
    if (seed) memcpy(out->grad, seed, sizeof(float) * (size_t)out->numel);
    else for (int64_t i = 0; i < out->numel; i++) out->grad[i] = 1.0f;

    /* Consumers before producers: build_postorder guarantees that order, so a
     * node is only visited after every consumer has contributed to its grad. */
    for (size_t i = 0; i < post.n; i++) {
        Tensor *node = post.items[i];
        Tensor *p0 = node->parent[0];
        Tensor *p1 = node->nparents > 1 ? node->parent[1] : NULL;
        const OpSpec *sp = &g_ops[node->op];

        /* OP_SUM produces a rank-1 tensor, so a loss node has exactly one
         * parent; the guard is nevertheless on "has grad and something to
         * propagate to", not on nparents, so a parentless output still gets
         * its seeded gradient. */
        if (node->requires_grad && node->grad && node->nparents > 0) {
            /* Each rule writes straight into the parents' grad buffers, so
             * every gradient-taking parent needs one first -- including the
             * unary ops, which is easy to get wrong. */
            for (int k = 0; k < node->nparents; k++) {
                Tensor *p = node->parent[k];
                if (p && p->requires_grad && !p->freed && !p->grad) grad_ensure(p);
            }

            if (p0 && p0 == p1) {
                /* e.g. tensor_add(a, a): both edges of the SAME tensor carry
                 * gradient, so apply the rule once into a temp and add it
                 * once, instead of feeding the shared grad buffer in twice. */
                float *tmp = (float *)calloc((size_t)(p0->numel > 0 ? p0->numel : 1),
                                             sizeof(float));
                if (sp->bwd2) sp->bwd2(node, tmp, tmp, node->grad);
                else if (sp->bwd1) sp->bwd1(node, 0, tmp, node->grad);
                for (int64_t q = 0; q < p0->numel; q++) p0->grad[q] += tmp[q];
                free(tmp);
            } else {
                float *ga = (p0 && p0->requires_grad && !p0->freed) ? p0->grad : NULL;
                float *gb = (p1 && p1->requires_grad && !p1->freed) ? p1->grad : NULL;
                if (sp->bwd2) sp->bwd2(node, ga, gb, node->grad);
                else if (sp->bwd1) {
                    if (ga) sp->bwd1(node, 0, ga, node->grad);
                    if (gb) sp->bwd1(node, 1, gb, node->grad);
                }
            }
        }
    }
    free(post.items);
}
/* ================================================================== */
/* Numeric gradient helper (tests)                                    */
/* ================================================================== */

/* Central differences on a scalar loss.
 *
 * Each evaluation runs on its own probe tensor, so the caller's `param` is
 * never part of the graph that gets freed here. */
float tensor_numeric_grad(Tensor *param, const int64_t *idx, float eps,
                          Tensor *(*f)(Tensor *param, void *ud), void *ud) {
    int64_t off = flat_index(param, idx);
    float orig = param->data[off];
    float plus, minus;

    for (int pass = 0; pass < 2; pass++) {
        Tensor *probe = tensor_clone(param);
        probe->requires_grad = 1;
        probe->held = 1;
        probe->data[off] = orig + (pass == 0 ? eps : -eps);
        Tensor *loss = f(probe, ud);
        if (!loss || loss->numel < 1) {
            fprintf(stderr, "[autograd] numeric grad: f() returned no value\n");
            tensor_free(loss);
            tensor_free(probe);
            return 0.0f;
        }
        tensor_backward(loss, NULL);   /* also detaches the graph from roots */
        if (pass == 0) plus = loss->data[0];
        else minus = loss->data[0];
        tensor_free(loss);
        tensor_free(probe);
    }

    return (plus - minus) / (2.0f * eps);
}

/* ================================================================== */
/* Printing                                                           */
/* ================================================================== */

static void print_rec(const Tensor *t, int64_t base, int dim) {
    if (dim == t->ndim - 1) {
        printf("[");
        for (int64_t i = 0; i < t->shape[dim]; i++) {
            printf("%.4f", t->data[base + i * t->stride[dim]]);
            if (i + 1 < t->shape[dim]) printf(", ");
        }
        printf("]");
        return;
    }
    printf("[");
    for (int64_t i = 0; i < t->shape[dim]; i++) {
        print_rec(t, base + i * t->stride[dim], dim + 1);
        if (i + 1 < t->shape[dim]) printf(", ");
    }
    printf("]");
}

void tensor_print(const Tensor *t, const char *label) {
    if (!t) { printf("%s = <null>\n", label ? label : "tensor"); return; }
    if (label) printf("%s = ", label);
    if (t->ndim == 0) { printf("%.4f", t->data[0]); }
    else print_rec(t, 0, 0);
    printf("  [shape=(");
    for (int d = 0; d < t->ndim; d++) printf("%lld%s", (long long)t->shape[d],
                                              d + 1 < t->ndim ? "," : "");
    printf("), op=%s]\n", tensor_op_name(t));
}
