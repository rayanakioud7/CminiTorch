/* data.h -- tiny dataset containers and synthetic datasets.
 *
 * Everything is float tensors already in memory: the point of this project is
 * the autograd and the layers, not the IO.
 */
#ifndef MINITORCH_DATA_H
#define MINITORCH_DATA_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Deterministic RNG shared by all the demos. */
void               mt_seed(unsigned long long seed);
unsigned long long mt_rng_state(void);
unsigned long long *mt_rng(void);

typedef struct {
    Tensor *inputs;    /* [N, ...] */
    Tensor *targets;   /* [N] integer labels, or [N, ...] floats */
    int64_t n;
} Dataset;

/* Shuffle in place and hand out consecutive slices. */
typedef struct {
    Dataset *ds;
    int64_t  batch_size;
    int64_t  index;      /* next sample to hand out */
    int64_t *order;      /* permutation */
} DataLoader;

Dataset    *dataset_create(MiniTorchScope *s, int64_t n, const int64_t *in_shape, int in_ndim,
                           const int64_t *tgt_shape, int tgt_ndim);
void        dataset_free(Dataset *ds);
DataLoader *loader_create(Dataset *ds, int64_t batch_size);
void        loader_free(DataLoader *dl);
/* Copies the next batch into freshly allocated tensors (caller frees them). */
int         loader_next(DataLoader *dl, Tensor **batch_in, Tensor **batch_tgt);
void        loader_reset(DataLoader *dl, int shuffle);

/* ---- synthetic problems ------------------------------------------ */

/* XOR: 4 points, 2 classes. A linear model cannot fit this. */
Dataset *dataset_xor(MiniTorchScope *s);
/* Two interleaved spirals -- the classic non-linear MLP benchmark. */
Dataset *dataset_spiral(MiniTorchScope *s, int64_t per_class, float noise);
/* 8x8 images: class 0 = vertical bar, class 1 = horizontal bar. */
Dataset *dataset_bars(MiniTorchScope *s, int64_t n_per_class, int size);
/* Character-level sequences over a tiny vocabulary (see dataset_text). */
Dataset *dataset_text(MiniTorchScope *s, const char *text, int64_t seq_len);

/* A short synthetic corpus with learnable structure (repeating motifs). */
const char *synthetic_corpus(void);

#ifdef __cplusplus
}
#endif

#endif /* MINITORCH_DATA_H */
