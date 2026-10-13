/* data.c -- RNG, dataset containers and the synthetic problems. */
#include "data.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================================================================== */
/* RNG                                                                */
/* ================================================================== */

static unsigned long long g_rng = 0x9E3779B97F4A7C15ULL;

unsigned long long *mt_rng(void);

/* Publishing the stream on first use keeps layer constructors (which take a
 * NULL rng) in sync with this file's generator. */
void mt_seed(unsigned long long seed) {
    g_rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
    mt_default_rng = mt_rng;
}

unsigned long long mt_rng_state(void) { return g_rng; }

unsigned long long *mt_rng(void) {
    mt_default_rng = mt_rng;
    return &g_rng;
}

static float frand(void) { return mt_rand_uniform(&g_rng); }

/* ================================================================== */
/* Dataset / DataLoader                                               */
/* ================================================================== */

Dataset *dataset_create(MiniTorchScope *s, int64_t n, const int64_t *in_shape, int in_ndim,
                        const int64_t *tgt_shape, int tgt_ndim) {
    Dataset *ds = (Dataset *)calloc(1, sizeof(Dataset));
    ds->n = n;
    int64_t ish[4], tsh[4];
    ish[0] = n;
    for (int i = 0; i < in_ndim; i++) ish[i + 1] = in_shape[i];
    ds->inputs = tensor_zeros(s, ish, in_ndim + 1);

    if (tgt_ndim == 1 && tgt_shape == NULL) {
        tsh[0] = n;
        ds->targets = tensor_zeros(s, tsh, 1);
    } else {
        tsh[0] = n;
        for (int i = 0; i < tgt_ndim; i++) tsh[i + 1] = tgt_shape[i];
        ds->targets = tensor_zeros(s, tsh, tgt_ndim + 1);
    }
    return ds;
}

void dataset_free(Dataset *ds) { free(ds); }

DataLoader *loader_create(Dataset *ds, int64_t batch_size) {
    DataLoader *dl = (DataLoader *)calloc(1, sizeof(DataLoader));
    dl->ds = ds;
    dl->batch_size = batch_size > 0 ? batch_size : 1;
    dl->order = (int64_t *)malloc(sizeof(int64_t) * (size_t)(ds->n > 0 ? ds->n : 1));
    for (int64_t i = 0; i < ds->n; i++) dl->order[i] = i;
    loader_reset(dl, 1);
    return dl;
}

void loader_free(DataLoader *dl) {
    if (!dl) return;
    free(dl->order);
    free(dl);
}

void loader_reset(DataLoader *dl, int shuffle) {
    dl->index = 0;
    if (shuffle) {
        for (int64_t i = dl->ds->n - 1; i > 0; i--) {
            int64_t j = (int64_t)(frand() * (float)(i + 1));
            if (j > i) j = i;
            int64_t tmp = dl->order[i];
            dl->order[i] = dl->order[j];
            dl->order[j] = tmp;
        }
    }
}

int loader_next(DataLoader *dl, Tensor **batch_in, Tensor **batch_tgt) {
    if (dl->index >= dl->ds->n) return 0;
    int64_t take = dl->batch_size;
    if (dl->index + take > dl->ds->n) take = dl->ds->n - dl->index;

    Dataset *ds = dl->ds;
    int64_t iw = ds->inputs->numel / ds->n;
    int64_t tw = ds->targets->numel / ds->n;
    int64_t ish[4] = { take, iw };
    int64_t tsh[4] = { take, tw };
    Tensor *bi = tensor_zeros(ds->inputs->scope, ish, 2);
    Tensor *bt = tensor_zeros(ds->targets->scope, tsh, 2);

    for (int64_t k = 0; k < take; k++) {
        int64_t row = dl->order[dl->index + k];
        memcpy(bi->data + k * iw, ds->inputs->data + row * iw, sizeof(float) * (size_t)iw);
        memcpy(bt->data + k * tw, ds->targets->data + row * tw, sizeof(float) * (size_t)tw);
    }
    dl->index += take;
    if (batch_in) *batch_in = bi;
    if (batch_tgt) *batch_tgt = bt;
    return (int)take;
}

/* ================================================================== */
/* XOR                                                                */
/* ================================================================== */

Dataset *dataset_xor(MiniTorchScope *s) {
    Dataset *ds = dataset_create(s, 4, (int64_t[]){ 2 }, 1, NULL, 1);
    float x[4][2] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
    float y[4] = { 0, 1, 1, 0 };
    for (int i = 0; i < 4; i++) {
        ds->inputs->data[i * 2 + 0] = x[i][0];
        ds->inputs->data[i * 2 + 1] = x[i][1];
        ds->targets->data[i] = y[i];
    }
    return ds;
}

/* ================================================================== */
/* Two spirals                                                         */
/* ================================================================== */

Dataset *dataset_spiral(MiniTorchScope *s, int64_t per_class, float noise) {
    int64_t n = per_class * 2;
    Dataset *ds = dataset_create(s, n, (int64_t[]){ 2 }, 1, NULL, 1);
    for (int64_t c = 0; c < 2; c++) {
        for (int64_t i = 0; i < per_class; i++) {
            float r = (float)i / (float)per_class;
            float t = 1.75f * r * 2.0f * (float)M_PI + (float)c * (float)M_PI;
            float x = r * cosf(t) + noise * (frand() - 0.5f);
            float y = r * sinf(t) + noise * (frand() - 0.5f);
            int64_t row = c * per_class + i;
            ds->inputs->data[row * 2 + 0] = x;
            ds->inputs->data[row * 2 + 1] = y;
            ds->targets->data[row] = (float)c;
        }
    }
    return ds;
}

/* ================================================================== */
/* Bar images: 8x8, class 0 = vertical bar, class 1 = horizontal bar   */
/* ================================================================== */

Dataset *dataset_bars(MiniTorchScope *s, int64_t n_per_class, int size) {
    int64_t n = n_per_class * 2;
    Dataset *ds = dataset_create(s, n, (int64_t[]){ 1, size, size }, 3, NULL, 1);
    for (int64_t c = 0; c < 2; c++) {
        for (int64_t i = 0; i < n_per_class; i++) {
            int64_t row = c * n_per_class + i;
            float *img = ds->inputs->data + row * size * size;
            int pos = (int)(frand() * (float)size);
            if (pos >= size) pos = size - 1;
            int width = 1 + (int)(frand() * 1.99f);   /* bar thickness 1..2 */
            for (int a = 0; a < size; a++)
                for (int b = 0; b < size; b++) {
                    int on;
                    if (c == 0) on = (b >= pos && b < pos + width);
                    else        on = (a >= pos && a < pos + width);
                    img[a * size + b] = on ? 1.0f : 0.0f;
                }
            /* a little noise so it is not trivially separable */
            for (int k = 0; k < size * size / 16; k++) {
                int idx = (int)(frand() * (float)(size * size));
                if (idx < size * size) img[idx] = 1.0f - img[idx];
            }
            ds->targets->data[row] = (float)c;
        }
    }
    return ds;
}

/* ================================================================== */
/* Character-level text                                                */
/* ================================================================== */

const char *synthetic_corpus(void) {
    /* Repetitive but not trivial: three motifs in a fixed rotation, so a
     * model must use more than one character of context to predict well. */
    static char buf[4096];
    const char *motifs[3] = { "abcabc", "deffde", "ghigih" };
    int len = 0;
    for (int rep = 0; rep < 60; rep++)
        for (int m = 0; m < 3; m++) {
            const char *p = motifs[m];
            while (*p && len < (int)sizeof(buf) - 2) buf[len++] = *p++;
        }
    buf[len] = '\0';
    return buf;
}

/* Builds (input, target) pairs of token ids: input = text[i..i+seq_len),
 * target = text[i+1..i+seq_len+1). Stored as float ids. */
Dataset *dataset_text(MiniTorchScope *s, const char *text, int64_t seq_len) {
    int64_t len = (int64_t)strlen(text);
    int64_t n = len - seq_len;                  /* one sample per window */
    if (n < 1) n = 1;
    Dataset *ds = dataset_create(s, n, (int64_t[]){ seq_len }, 1, (int64_t[]){ seq_len }, 1);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < seq_len; j++) {
            int64_t a = (i + j) % len;
            int64_t b = (i + j + 1) % len;
            ds->inputs->data[i * seq_len + j] = (float)(text[a] - 'a');
            ds->targets->data[i * seq_len + j] = (float)(text[b] - 'a');
        }
    }
    return ds;
}
