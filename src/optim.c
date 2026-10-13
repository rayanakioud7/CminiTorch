/* optim.c -- SGD and Adam.
 *
 * Each optimizer state buffer is one flat allocation per parameter slot, so
 * lookups are O(1) and freeing is a single call.
 */
#include "optim.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static Optimizer *optim_new(Tensor **params, int n) {
    Optimizer *o = (Optimizer *)calloc(1, sizeof(Optimizer));
    o->nparams = n;
    o->params = (Tensor **)malloc(sizeof(Tensor *) * (size_t)(n > 0 ? n : 1));
    memcpy(o->params, params, sizeof(Tensor *) * (size_t)n);
    /* make sure every parameter has a gradient buffer to read later */
    for (int i = 0; i < n; i++) {
        if (!o->params[i]->grad)
            o->params[i]->grad =
                (float *)calloc((size_t)(o->params[i]->numel > 0 ? o->params[i]->numel : 1),
                                sizeof(float));
    }
    return o;
}

Optimizer *optim_sgd(Tensor **params, int n, float lr, float momentum, float weight_decay) {
    Optimizer *o = optim_new(params, n);
    o->kind = OPT_SGD;
    o->lr = lr;
    o->momentum = momentum;
    o->weight_decay = weight_decay;
    if (momentum > 0.0f) {
        o->velocity = (float **)calloc((size_t)(n > 0 ? n : 1), sizeof(float *));
        for (int i = 0; i < n; i++) {
            int64_t k = params[i]->numel > 0 ? params[i]->numel : 1;
            o->velocity[i] = (float *)calloc((size_t)k, sizeof(float));
        }
    }
    return o;
}

Optimizer *optim_adam(Tensor **params, int n, float lr, float beta1, float beta2, float eps) {
    Optimizer *o = optim_new(params, n);
    o->kind = OPT_ADAM;
    o->lr = lr;
    o->beta1 = beta1;
    o->beta2 = beta2;
    o->eps = eps;
    o->m = (float **)calloc((size_t)(n > 0 ? n : 1), sizeof(float *));
    o->v = (float **)calloc((size_t)(n > 0 ? n : 1), sizeof(float *));
    for (int i = 0; i < n; i++) {
        int64_t k = params[i]->numel > 0 ? params[i]->numel : 1;
        o->m[i] = (float *)calloc((size_t)k, sizeof(float));
        o->v[i] = (float *)calloc((size_t)k, sizeof(float));
    }
    return o;
}

void optim_free(Optimizer *o) {
    if (!o) return;
    if (o->velocity) {
        for (int i = 0; i < o->nparams; i++) free(o->velocity[i]);
        free(o->velocity);
    }
    if (o->m) {
        for (int i = 0; i < o->nparams; i++) free(o->m[i]);
        free(o->m);
    }
    if (o->v) {
        for (int i = 0; i < o->nparams; i++) free(o->v[i]);
        free(o->v);
    }
    free(o->params);
    free(o);
}

void optim_zero_grad(Optimizer *o) {
    for (int i = 0; i < o->nparams; i++) tensor_zero_grad(o->params[i]);
}

void optim_step(Optimizer *o) {
    if (o->kind == OPT_SGD) {
        float **vel = o->velocity;
        for (int i = 0; i < o->nparams; i++) {
            Tensor *p = o->params[i];
            if (!p->grad) continue;
            for (int64_t k = 0; k < p->numel; k++) {
                float g = p->grad[k];
                if (o->weight_decay > 0.0f) g += o->weight_decay * p->data[k];
                if (vel) {
                    vel[i][k] = o->momentum * vel[i][k] + (1.0f - o->dampening) * g;
                    g = vel[i][k];
                }
                p->data[k] -= o->lr * g;
            }
        }
        return;
    }

    /* Adam */
    o->step_count++;
    float b1 = o->beta1, b2 = o->beta2;
    float bc1 = 1.0f - powf(b1, (float)o->step_count);
    float bc2 = 1.0f - powf(b2, (float)o->step_count);
    for (int i = 0; i < o->nparams; i++) {
        Tensor *p = o->params[i];
        if (!p->grad) continue;
        for (int64_t k = 0; k < p->numel; k++) {
            float g = p->grad[k];
            o->m[i][k] = b1 * o->m[i][k] + (1.0f - b1) * g;
            o->v[i][k] = b2 * o->v[i][k] + (1.0f - b2) * g * g;
            float mhat = o->m[i][k] / bc1;
            float vhat = o->v[i][k] / bc2;
            p->data[k] -= o->lr * mhat / (sqrtf(vhat) + o->eps);
        }
    }
}

float sched_step(float *lr, float gamma, int every, int step) {
    if (every > 0 && step > 0 && step % every == 0) *lr *= gamma;
    return *lr;
}

float sched_cosine(float lr0, float lr_min, int step, int total) {
    if (total <= 0 || step >= total) return lr_min;
    float t = (float)step / (float)total;
    return lr_min + 0.5f * (lr0 - lr_min) * (1.0f + cosf(3.14159265f * t));
}
