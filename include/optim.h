/* optim.h -- SGD (with momentum) and Adam.
 *
 * An optimizer holds a flat list of parameter tensors. The graph was built
 * against those same tensors, so after tensor_backward() their ->grad fields
 * hold dL/dp and step() can update them in place.
 */
#ifndef MINITORCH_OPTIM_H
#define MINITORCH_OPTIM_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { OPT_SGD = 0, OPT_ADAM = 1 } OptimKind;

typedef struct {
    OptimKind kind;
    Tensor  **params;
    int       nparams;

    float lr;
    /* SGD */
    float momentum;
    float weight_decay;
    float dampening;
    float **velocity;     /* one buffer per parameter (momentum) */
    /* Adam */
    float beta1, beta2, eps;
    float **m, **v;       /* first and second moment buffers per parameter */
    int    step_count;
} Optimizer;

/* Build an optimizer over `params` (the array is copied). */
Optimizer *optim_sgd(Tensor **params, int n, float lr, float momentum, float weight_decay);
Optimizer *optim_adam(Tensor **params, int n, float lr, float beta1, float beta2, float eps);
void       optim_free(Optimizer *o);
void       optim_step(Optimizer *o);
void       optim_zero_grad(Optimizer *o);

/* ---- learning-rate schedules ------------------------------------- */
float sched_step(float *lr, float gamma, int every, int step);
float sched_cosine(float lr0, float lr_min, int step, int total);

#ifdef __cplusplus
}
#endif

#endif /* MINITORCH_OPTIM_H */
