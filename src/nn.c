/* nn.c -- modules, layers and the transformer block.
 *
 * Modules form a tree and parameters are gathered by walking it. Everything
 * differentiable is expressed with ops from tensor.c, so the autograd graph is
 * produced entirely by the forward pass.
 */
#include "nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================== */
/* Module plumbing                                                    */
/* ================================================================== */

static void module_free_self(Module *self);

void module_init(Module *m, const ModuleVTable *vt, const char *name) {
    memset(m, 0, sizeof(*m));
    m->vt = vt;
    snprintf(m->name, sizeof(m->name), "%s", name ? name : "module");
    m->training = 1;
}

void module_add_param(Module *m, Tensor *p) {
    if (!p) return;
    p->held = 1;
    if (m->nparams == m->params_cap) {
        m->params_cap = m->params_cap ? m->params_cap * 2 : 8;
        m->params = (Tensor **)realloc(m->params, sizeof(Tensor *) * (size_t)m->params_cap);
    }
    m->params[m->nparams++] = p;
}

void module_add_child(Module *m, Module *child) {
    if (!child) return;
    if (m->nchildren == m->children_cap) {
        m->children_cap = m->children_cap ? m->children_cap * 2 : 8;
        m->children = (Module **)realloc(m->children, sizeof(Module *) * (size_t)m->children_cap);
    }
    m->children[m->nchildren++] = child;
}

Tensor *module_forward(Module *m, Tensor *input) {
    if (!m) return input;
    m->output = NULL;
    if (m->vt && m->vt->forward) m->vt->forward(m, input);
    return m->output;
}

Tensor *module_apply(Module *m, Tensor *input) {
    if (!m || !m->vt) return input;
    if (m->vt->forward) return module_forward(m, input);
    Tensor *cur = input;
    for (int i = 0; i < m->nchildren; i++) cur = module_apply(m->children[i], cur);
    m->output = cur;
    return cur;
}

void module_set_training(Module *m, int training) {
    if (!m) return;
    m->training = training;
    for (int i = 0; i < m->nchildren; i++) module_set_training(m->children[i], training);
}

void module_zero_grad(Module *m) {
    if (!m) return;
    for (int i = 0; i < m->nparams; i++) tensor_zero_grad(m->params[i]);
    for (int i = 0; i < m->nchildren; i++) module_zero_grad(m->children[i]);
}

static void collect_params(Module *m, Tensor ***out, int *n, int *cap) {
    for (int i = 0; i < m->nparams; i++) {
        if (out) {
            if (*n == *cap) {
                *cap = *cap ? *cap * 2 : 16;
                *out = (Tensor **)realloc(*out, sizeof(Tensor *) * (size_t)*cap);
            }
            (*out)[(*n)++] = m->params[i];
        } else {
            (*n)++;
        }
    }
    for (int i = 0; i < m->nchildren; i++) collect_params(m->children[i], out, n, cap);
}

Tensor **module_params(Module *m, int *count) {
    Tensor **out = NULL;
    int n = 0, cap = 0;
    collect_params(m, &out, &n, &cap);
    if (count) *count = n;
    return out;
}

int module_num_params(Module *m) {
    int n = 0;
    collect_params(m, NULL, &n, NULL);
    return n;
}

void module_free(Module *m) {
    if (!m) return;
    for (int i = 0; i < m->nchildren; i++) module_free(m->children[i]);
    /* Release the bookkeeping BEFORE the struct itself, and never touch `m`
     * afterwards. (Each module is one allocation, so the vtable does not need
     * a custom destructor.) */
    free(m->params);
    free(m->children);
    if (m->vt && m->vt->free && m->vt->free != module_free_self) m->vt->free(m);
    else free(m);
}

void module_print(Module *m, int indent) {
    if (!m) return;
    for (int i = 0; i < indent; i++) printf("  ");
    printf("%s", m->name);
    int64_t total = 0;
    for (int i = 0; i < m->nparams; i++) total += m->params[i]->numel;
    if (total) printf("  (%lld params)", (long long)total);
    printf("\n");
    for (int i = 0; i < m->nchildren; i++) module_print(m->children[i], indent + 1);
}

/* ================================================================== */
/* Linear                                                            */
/* ================================================================== */

static void linear_forward(Module *self, Tensor *in) {
    Linear *l = (Linear *)self;
    int last = in->ndim - 1;
    int64_t in_f = in->shape[last];
    int64_t rows = in->numel / (in_f > 0 ? in_f : 1);

    Tensor *flat = tensor_reshape(in, (int64_t[]){ rows, in_f }, 2);
    Tensor *out = tensor_matmul_tn(flat, l->weight);      /* [rows, out_features] */
    if (l->bias) out = tensor_add(out, l->bias);          /* broadcast over rows */

    int64_t shape[4];
    for (int i = 0; i < in->ndim - 1; i++) shape[i] = in->shape[i];
    shape[in->ndim - 1] = l->weight->shape[0];
    self->output = tensor_reshape(out, shape, in->ndim);
}

/* Every module is one allocation owned by its creator, so the generic
 * teardown in module_free() is enough and this hook exists only to keep the
 * vtable shape explicit. */
static void module_free_self(Module *self) { free(self); }
static const ModuleVTable LINEAR_VT = { "Linear", linear_forward, module_free_self };

Tensor *init_linear_weight(MiniTorchScope *s, int out, int in, unsigned long long *rng) {
    int64_t sh[2] = { out, in };
    Tensor *w = tensor_rand_normal(s, sh, 2, 1.0f / sqrtf((float)in), rng);
    w->requires_grad = 1;
    return w;
}

Linear *linear_create(MiniTorchScope *s, int in_features, int out_features, int bias) {
    Linear *l = (Linear *)calloc(1, sizeof(Linear));
    module_init(&l->base, &LINEAR_VT, "Linear");
    l->weight = init_linear_weight(s, out_features, in_features, NULL);
    module_add_param(&l->base, l->weight);
    if (bias) {
        int64_t bs[1] = { out_features };
        l->bias = tensor_zeros(s, bs, 1);
        l->bias->requires_grad = 1;
        module_add_param(&l->base, l->bias);
    }
    return l;
}

/* ================================================================== */
/* Activations and shape ops (no parameters)                          */
/* ================================================================== */

typedef struct { Module base; } PlainModule;

static void relu_forward(Module *self, Tensor *in) { self->output = tensor_relu(in); }
static void gelu_forward(Module *self, Tensor *in) { self->output = tensor_gelu(in); }

static void flatten_forward(Module *self, Tensor *in) {
    int64_t batch = in->shape[0];
    self->output = tensor_reshape(in, (int64_t[]){ batch, in->numel / batch }, 2);
}

static const ModuleVTable RELU_VT = { "ReLU", relu_forward, module_free_self };
static const ModuleVTable GELU_VT = { "GELU", gelu_forward, module_free_self };
static const ModuleVTable FLATTEN_VT = { "Flatten", flatten_forward, module_free_self };

static Module *plain_create(const ModuleVTable *vt, const char *name) {
    PlainModule *p = (PlainModule *)calloc(1, sizeof(PlainModule));
    module_init(&p->base, vt, name);
    return &p->base;
}

Module *relu_create(void) { return plain_create(&RELU_VT, "ReLU"); }
Module *gelu_create(void) { return plain_create(&GELU_VT, "GELU"); }
Module *flatten_create(void) { return plain_create(&FLATTEN_VT, "Flatten"); }

/* ================================================================== */
/* Sequential                                                        */
/* ================================================================== */

static void seq_forward(Module *self, Tensor *in) {
    Tensor *cur = in;
    for (int i = 0; i < self->nchildren; i++) cur = module_apply(self->children[i], cur);
    self->output = cur;
}
static const ModuleVTable SEQ_VT = { "Sequential", seq_forward, module_free_self };

Sequential *sequential_create(void) {
    Sequential *s = (Sequential *)calloc(1, sizeof(Sequential));
    module_init(&s->base, &SEQ_VT, "Sequential");
    return s;
}

void sequential_add(Sequential *seq, Module *child) { module_add_child(&seq->base, child); }

/* ================================================================== */
/* Conv2d / MaxPool2d                                                 */
/* ================================================================== */

static void conv_forward(Module *self, Tensor *in) {
    Conv2d *c = (Conv2d *)self;
    self->output = tensor_conv2d(in, c->weight, c->bias, c->stride, c->padding, c->dilation);
}
static const ModuleVTable CONV_VT = { "Conv2d", conv_forward, module_free_self };

Conv2d *conv2d_create(MiniTorchScope *s, int in_ch, int out_ch, int k,
                      int stride, int padding, int dilation, int bias) {
    Conv2d *c = (Conv2d *)calloc(1, sizeof(Conv2d));
    module_init(&c->base, &CONV_VT, "Conv2d");
    c->stride = stride;
    c->padding = padding;
    c->dilation = dilation;
    int64_t ws[4] = { out_ch, in_ch, k, k };
    float std = sqrtf(2.0f / (float)(in_ch * k * k));   /* Kaiming-ish */
    c->weight = tensor_rand_normal(s, ws, 4, std, NULL);
    c->weight->requires_grad = 1;
    module_add_param(&c->base, c->weight);
    if (bias) {
        int64_t bs[1] = { out_ch };
        c->bias = tensor_zeros(s, bs, 1);
        c->bias->requires_grad = 1;
        module_add_param(&c->base, c->bias);
    }
    return c;
}

static void pool_forward(Module *self, Tensor *in) {
    MaxPool2d *p = (MaxPool2d *)self;
    self->output = tensor_maxpool2d(in, p->k, p->stride);
}
static const ModuleVTable POOL_VT = { "MaxPool2d", pool_forward, module_free_self };

MaxPool2d *maxpool2d_create(int k, int stride) {
    MaxPool2d *p = (MaxPool2d *)calloc(1, sizeof(MaxPool2d));
    module_init(&p->base, &POOL_VT, "MaxPool2d");
    p->k = k;
    p->stride = stride > 0 ? stride : k;
    return p;
}

/* ================================================================== */
/* LayerNorm over the last dimension                                  */
/* ================================================================== */

static void layernorm_forward(Module *self, Tensor *in) {
    LayerNorm1d *ln = (LayerNorm1d *)self;
    int last = in->ndim - 1;

    Tensor *mean = tensor_mean_dim(in, last, 1);          /* [.., 1] */
    Tensor *centered = tensor_sub(in, mean);
    Tensor *var = tensor_mean_dim(tensor_mul(centered, centered), last, 1);
    Tensor *denom = tensor_sqrt(tensor_add_scalar(var, ln->eps));
    Tensor *normed = tensor_div(centered, denom);
    self->output = tensor_add(tensor_mul(normed, ln->gamma), ln->beta);
}
static const ModuleVTable LN_VT = { "LayerNorm", layernorm_forward, module_free_self };

LayerNorm1d *layernorm1d_create(MiniTorchScope *s, int features) {
    LayerNorm1d *ln = (LayerNorm1d *)calloc(1, sizeof(LayerNorm1d));
    module_init(&ln->base, &LN_VT, "LayerNorm");
    ln->features = features;
    ln->eps = 1e-5f;
    int64_t sh[1] = { features };
    ln->gamma = tensor_full(s, sh, 1, 1.0f);
    ln->gamma->requires_grad = 1;
    ln->beta = tensor_zeros(s, sh, 1);
    ln->beta->requires_grad = 1;
    module_add_param(&ln->base, ln->gamma);
    module_add_param(&ln->base, ln->beta);
    return ln;
}

/* ================================================================== */
/* Embedding                                                          */
/* ================================================================== */

/* Driven by embedding_forward() because the input is integer ids. */
static void embedding_noop(Module *self, Tensor *in) { (void)self; (void)in; }
static const ModuleVTable EMB_VT = { "Embedding", embedding_noop, module_free_self };

Embedding *embedding_create(MiniTorchScope *s, int vocab, int dim) {
    Embedding *e = (Embedding *)calloc(1, sizeof(Embedding));
    module_init(&e->base, &EMB_VT, "Embedding");
    e->vocab = vocab;
    e->dim = dim;
    int64_t sh[2] = { vocab, dim };
    e->weight = tensor_rand_normal(s, sh, 2, 0.02f, NULL);
    e->weight->requires_grad = 1;
    module_add_param(&e->base, e->weight);
    return e;
}

Tensor *embedding_forward(Embedding *e, const int64_t *ids, int64_t n_ids) {
    return tensor_embedding(e->weight, ids, n_ids);
}

/* ================================================================== */
/* Dropout                                                            */
/* ================================================================== */

static void dropout_forward(Module *self, Tensor *in) {
    Dropout *d = (Dropout *)self;
    self->output = tensor_dropout(in, d->p, self->training, d->rng);
}
static const ModuleVTable DROP_VT = { "Dropout", dropout_forward, module_free_self };

Dropout *dropout_create(float p, unsigned long long *rng) {
    Dropout *d = (Dropout *)calloc(1, sizeof(Dropout));
    module_init(&d->base, &DROP_VT, "Dropout");
    d->p = p;
    d->rng = rng;
    return d;
}

/* ================================================================== */
/* Losses                                                             */
/* ================================================================== */

Tensor *loss_cross_entropy_labels(Tensor *logits, const int64_t *labels, int64_t n) {
    if (logits->ndim != 2 || logits->shape[0] != n) {
        fprintf(stderr, "[nn] cross_entropy: logits must be [n, C]\n");
        return NULL;
    }
    int64_t C = logits->shape[1];
    Tensor *onehot = tensor_zeros(logits->scope, logits->shape, 2);
    for (int64_t i = 0; i < n; i++) {
        int64_t c = labels[i];
        if (c >= 0 && c < C) onehot->data[i * C + c] = 1.0f;
    }
    return tensor_cross_entropy(logits, onehot, 1);
}

/* ================================================================== */
/* Transformer block: pre-norm, multi-head self-attention, MLP        */
/* ================================================================== */

/* Split [T, H*dh] into H separate [T, dh] tensors.
 *
 * There is no autograd rule for slicing, so each head is extracted with a
 * matmul against a fixed 0/1 selector matrix S_h ([H*dh, dh]). That keeps the
 * whole thing differentiable with the ops the engine already has. */
static Tensor *head_slice(Tensor *x, MiniTorchScope *s, int64_t H, int64_t dh, int64_t h) {
    int64_t sh[2] = { H * dh, dh };
    Tensor *sel = tensor_zeros(s, sh, 2);
    for (int64_t i = 0; i < dh; i++) sel->data[(h * dh + i) * dh + i] = 1.0f;
    return tensor_matmul(x, sel);
}

/* Concatenate H [T, dh] tensors into [T, H*dh] by summing them into disjoint
 * column blocks. Each head's output is placed with a transposed selector. */
static Tensor *head_place(Tensor *x, MiniTorchScope *s, int64_t H, int64_t dh, int64_t h) {
    int64_t sh[2] = { dh, H * dh };
    Tensor *sel = tensor_zeros(s, sh, 2);
    for (int64_t i = 0; i < dh; i++) sel->data[i * (H * dh) + (h * dh + i)] = 1.0f;
    return tensor_matmul(x, sel);
}

/* q,k,v: [T, dh]; mask: [T, T] additive (0 or -1e9). Returns [T, dh]. */
static Tensor *head_attention(Tensor *q, Tensor *k, Tensor *v, const Tensor *mask) {
    int64_t dh = q->shape[1];
    Tensor *scores = tensor_matmul_tn(q, k);
    if (!scores) {
        fprintf(stderr, "  [attn] matmul_tn failed: q=[%lld,%lld] k=[%lld,%lld]\n",
                (long long)q->shape[0], (long long)q->shape[1], (long long)k->shape[0],
                (long long)k->shape[1]);
        return NULL;
    }
    scores = tensor_mul_scalar(scores, 1.0f / sqrtf((float)dh));
    if (mask) scores = tensor_add(scores, (Tensor *)mask);
    Tensor *probs = tensor_softmax(scores);
    if (!probs) return NULL;
    return tensor_matmul(probs, v);
}

static void transformer_forward(Module *self, Tensor *in) {
    TransformerBlock *b = (TransformerBlock *)self;
    int64_t T = in->shape[0];
    int64_t dim = b->dim, H = b->n_heads, dh = dim / H;
    MiniTorchScope *s = in->scope;

    /* The stored mask covers the full block size; sequences can be shorter, so
     * carve out the [T, T] top-left corner. It is a constant, so building it
     * per forward pass costs nothing in graph terms. */
    Tensor *mask = NULL;
    if (T <= b->block_size) {
        int64_t msh[2] = { T, T };
        mask = tensor_zeros(s, msh, 2);
        for (int64_t i = 0; i < T; i++)
            for (int64_t j = 0; j < T; j++)
                mask->data[i * T + j] = b->mask->data[i * b->block_size + j];
    }

    /* ---- attention ---- */
    Tensor *h = module_forward(&b->ln1->base, in);
    Tensor *qkv = module_forward(&b->qkv->base, h);       /* [T, 3*dim] */

    int64_t sh3[2] = { 3 * dim, dim };
    Tensor *sq = tensor_zeros(s, sh3, 2);
    Tensor *sk = tensor_zeros(s, sh3, 2);
    Tensor *sv = tensor_zeros(s, sh3, 2);
    for (int64_t i = 0; i < dim; i++) {
        sq->data[i * dim + i] = 1.0f;
        sk->data[(dim + i) * dim + i] = 1.0f;
        sv->data[(2 * dim + i) * dim + i] = 1.0f;
    }
    Tensor *q = tensor_matmul(qkv, sq);
    Tensor *k = tensor_matmul(qkv, sk);
    Tensor *v = tensor_matmul(qkv, sv);

    Tensor *attn = NULL;
    for (int64_t hh = 0; hh < H; hh++) {
        Tensor *qh = head_slice(q, s, H, dh, hh);
        Tensor *kh = head_slice(k, s, H, dh, hh);
        Tensor *vh = head_slice(v, s, H, dh, hh);
        Tensor *oh = head_attention(qh, kh, vh, mask);
        Tensor *placed = head_place(oh, s, H, dh, hh);
        attn = attn ? tensor_add(attn, placed) : placed;
    }
    Tensor *proj = module_forward(&b->proj->base, attn);
    Tensor *x1 = tensor_add(in, proj);                    /* residual */

    /* ---- feed-forward ---- */
    Tensor *h2 = module_forward(&b->ln2->base, x1);
    Tensor *f = module_forward(&b->fc1->base, h2);
    f = tensor_gelu(f);
    f = module_forward(&b->fc2->base, f);
    self->output = tensor_add(x1, f);                     /* residual */
}
static const ModuleVTable TBLOCK_VT = { "TransformerBlock", transformer_forward, module_free_self };

TransformerBlock *transformer_block_create(MiniTorchScope *s, int dim, int n_heads,
                                           int block_size, unsigned long long *rng) {
    TransformerBlock *b = (TransformerBlock *)calloc(1, sizeof(TransformerBlock));
    module_init(&b->base, &TBLOCK_VT, "TransformerBlock");
    b->dim = dim;
    b->n_heads = n_heads;
    b->block_size = block_size;

    b->ln1 = layernorm1d_create(s, dim);
    b->ln2 = layernorm1d_create(s, dim);
    b->qkv = linear_create(s, dim, 3 * dim, 1);
    b->proj = linear_create(s, dim, dim, 1);
    b->fc1 = linear_create(s, dim, 4 * dim, 1);
    b->fc2 = linear_create(s, 4 * dim, dim, 1);
    module_add_child(&b->base, &b->ln1->base);
    module_add_child(&b->base, &b->ln2->base);
    module_add_child(&b->base, &b->qkv->base);
    module_add_child(&b->base, &b->proj->base);
    module_add_child(&b->base, &b->fc1->base);
    module_add_child(&b->base, &b->fc2->base);

    /* causal mask: additive bias, 0 where j <= i and -1e9 where j > i */
    int64_t msh[2] = { block_size, block_size };
    b->mask = tensor_zeros(s, msh, 2);
    for (int64_t i = 0; i < block_size; i++)
        for (int64_t j = 0; j < block_size; j++)
            b->mask->data[i * block_size + j] = (j <= i) ? 0.0f : -1e9f;
    (void)rng;
    return b;
}
