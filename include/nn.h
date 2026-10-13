/* nn.h -- Module system and layers built on the tensor/autograd core.
 *
 * A Module mirrors torch.nn.Module: it owns parameters, can be composed into
 * trees, and has a forward() that builds the autograd graph. Parameters are
 * collected by pointer, so an optimizer sees exactly the tensors it must
 * update.
 */
#ifndef MINITORCH_NN_H
#define MINITORCH_NN_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Module Module;

typedef struct {
    const char *name;
    void (*forward)(Module *self, Tensor *in);
    void (*free)(Module *self);
} ModuleVTable;

struct Module {
    const ModuleVTable *vt;
    char   name[64];
    Tensor **params;
    int      nparams;
    int      params_cap;
    Module **children;
    int      nchildren;
    int      children_cap;
    /* forward() stores its output here so the tree can be driven generically */
    Tensor  *output;
    int      training;   /* dropout / batchnorm behave differently in eval mode */
};

/* Call the module's forward; returns the same pointer it stored in ->output. */
Tensor *module_forward(Module *m, Tensor *input);
/* Depth-first forward through the whole tree starting at `m`. */
Tensor *module_apply(Module *m, Tensor *input);
void    module_set_training(Module *m, int training);
void    module_zero_grad(Module *m);
int     module_num_params(Module *m);          /* including children */
Tensor **module_params(Module *m, int *count); /* flattened, caller frees array */
void    module_free(Module *m);
void    module_print(Module *m, int indent);

/* registration helpers used by the layer constructors */
void module_init(Module *m, const ModuleVTable *vt, const char *name);
void module_add_param(Module *m, Tensor *p);
void module_add_child(Module *m, Module *child);

/* ------------------------------------------------------------------ */
/* Layers                                                             */
/* ------------------------------------------------------------------ */

typedef struct { Module base; Tensor *weight; Tensor *bias; } Linear;
typedef struct { Module base; Tensor *weight; Tensor *bias;
                 int stride, padding, dilation; } Conv2d;
typedef struct { Module base; int k, stride; } MaxPool2d;
typedef struct { Module base; int features; float eps; Tensor *gamma; Tensor *beta; } LayerNorm1d;

Linear     *linear_create(MiniTorchScope *s, int in_features, int out_features, int bias);
Conv2d     *conv2d_create(MiniTorchScope *s, int in_ch, int out_ch, int k,
                          int stride, int padding, int dilation, int bias);
MaxPool2d  *maxpool2d_create(int k, int stride);
LayerNorm1d *layernorm1d_create(MiniTorchScope *s, int features);
Module     *relu_create(void);
Module     *gelu_create(void);
Module     *flatten_create(void);   /* [N, ...] -> [N, prod] */

/* ------------------------------------------------------------------ */
/* Sequential container                                               */
/* ------------------------------------------------------------------ */

typedef struct { Module base; } Sequential;
Sequential *sequential_create(void);
void        sequential_add(Sequential *seq, Module *child);

/* ------------------------------------------------------------------ */
/* Embedding / dropout                                                */
/* ------------------------------------------------------------------ */

typedef struct { Module base; Tensor *weight; int vocab, dim; } Embedding;
Embedding *embedding_create(MiniTorchScope *s, int vocab, int dim);
/* ids: n_ids integer token ids -> [n_ids, dim] */
Tensor *embedding_forward(Embedding *e, const int64_t *ids, int64_t n_ids);

typedef struct { Module base; float p; unsigned long long *rng; } Dropout;
Dropout *dropout_create(float p, unsigned long long *rng);

/* ------------------------------------------------------------------ */
/* Transformer block (pre-norm, single head or multi-head)             */
/* ------------------------------------------------------------------ */

typedef struct {
    Module base;
    int   dim, n_heads, block_size;
    Linear *qkv;      /* dim -> 3*dim */
    Linear *proj;     /* dim -> dim */
    Linear *fc1, *fc2;
    LayerNorm1d *ln1, *ln2;
    Tensor *mask;     /* [block_size, block_size] causal mask, 0/1 */
    int    training;
} TransformerBlock;

TransformerBlock *transformer_block_create(MiniTorchScope *s, int dim, int n_heads,
                                           int block_size, unsigned long long *rng);

/* ------------------------------------------------------------------ */
/* Losses                                                             */
/* ------------------------------------------------------------------ */

/* Cross entropy from integer labels. Builds the one-hot targets internally
 * and returns a scalar tensor connected to `logits`. */
Tensor *loss_cross_entropy_labels(Tensor *logits, const int64_t *labels, int64_t n);

/* Convenience: gaussian init of shape [out, in] scaled by 1/sqrt(in). */
Tensor *init_linear_weight(MiniTorchScope *s, int out, int in, unsigned long long *rng);

#ifdef __cplusplus
}
#endif

#endif /* MINITORCH_NN_H */
