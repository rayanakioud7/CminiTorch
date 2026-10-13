/* test_nn.c -- finite-difference checks for the layer stack.
 *
 * test_autograd.c verifies individual ops. This file verifies that whole
 * models (Linear -> activation -> Linear, Conv2d+Pool, LayerNorm, and the
 * transformer block) produce gradients consistent with numerical
 * differentiation of their own loss. If an optimizer is being fed wrong
 * gradients, this is where it shows up.
 */
#include "data.h"
#include "nn.h"
#include "optim.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass = 0, g_fail = 0;

static void report(const char *name, double worst, double tol) {
    if (worst > tol) {
        printf("  [FAIL] %-34s worst rel err %.3e (tol %.0e)\n", name, worst, tol);
        g_fail++;
    } else {
        printf("  [ ok ] %-34s worst rel err %.3e\n", name, worst);
        g_pass++;
    }
}

/* Compare every parameter's analytic gradient against a central difference of
 * the loss. `loss_fn` receives the module and returns a scalar tensor. */
typedef Tensor *(*LossFn)(Module *m, void *ud);

static void fd_check_model(const char *name, Module *m, LossFn loss_fn, void *ud,
                          double tol, int max_samples) {
    int nparams = 0;
    Tensor **params = module_params(m, &nparams);

    /* Analytic gradients: one forward, one backward, snapshot the results. */
    module_zero_grad(m);
    Tensor *loss = loss_fn(m, ud);
    tensor_backward(loss, NULL);
    tensor_free(loss);

    float **ana = (float **)calloc((size_t)nparams, sizeof(float *));
    for (int i = 0; i < nparams; i++) {
        ana[i] = (float *)malloc(sizeof(float) * (size_t)params[i]->numel);
        for (int64_t k = 0; k < params[i]->numel; k++)
            ana[i][k] = params[i]->grad ? params[i]->grad[k] : 0.0f;
    }

    /* Numerical gradients. The perturbation lives in the parameter itself, so
     * it must be restored from a saved scalar before any graph that references
     * it is freed -- reading it back afterwards would be a use-after-free. */
    double worst = 0.0;
    int checked = 0;
    int wi = -1, wd = -1;
    int64_t wk = -1;
    double wnum = 0.0, wana = 0.0, wplus = 0.0, wminus = 0.0;
    for (int i = 0; i < nparams && checked < max_samples; i++) {
        int64_t n = params[i]->numel;
        for (int s = 0; s < 3 && checked < max_samples; s++) {
            int64_t k = (int64_t)((double)s * (double)(n - 1) / 3.0);
            if (k < 0 || k >= n) continue;
            float orig = params[i]->data[k];
            float plus, minus;

            for (int pass = 0; pass < 2; pass++) {
                params[i]->data[k] = orig + (pass == 0 ? 1e-3f : -1e-3f);
                Tensor *l = loss_fn(m, ud);
                if (pass == 0) plus = l->data[0];
                else minus = l->data[0];
                tensor_free(l);
            }
            params[i]->data[k] = orig;

            double num = (double)(plus - minus) / 2e-3;
            double a = ana[i][k];
            double denom = fabs(a) > 1e-3 ? fabs(a) : 1.0;
            double rel = fabs(a - num) / denom;
            if (rel > worst) {
                worst = rel;
                wi = i; wk = k; wnum = num; wana = a; wplus = plus; wminus = minus; wd = 0;
            }
            checked++;
        }
    }
    if (worst > tol && wd == 0)
        printf("      [diag] %s: param%d[%lld] ana=%.6f num=%.6f (L+=%.6f L-=%.6f) rel=%.2e\n",
               name, wi, (long long)wk, wana, wnum, wplus, wminus, worst);

    for (int i = 0; i < nparams; i++) free(ana[i]);
    free(ana);
    free(params);
    report(name, worst, tol);
}

/* ---- fixtures ---------------------------------------------------- */

typedef struct { Tensor *x; Tensor *labels; int64_t lab[8]; } ClsArg;

static Tensor *mlp_loss(Module *m, void *ud) {
    ClsArg *a = (ClsArg *)ud;
    Tensor *out = module_apply(m, a->x);
    return loss_cross_entropy_labels(out, a->lab, out->shape[0]);
}

static Tensor *sse_loss(Module *m, void *ud) {
    ClsArg *a = (ClsArg *)ud;
    Tensor *out = module_apply(m, a->x);
    return tensor_sum(tensor_mul(out, out));   /* smooth, no class boundaries */
}

/* ---- tests ------------------------------------------------------- */

static void test_mlp(void) {
    MiniTorchScope *s = scope_create();
    unsigned long long rng = 99;
    int64_t xsh[2] = { 6, 3 };
    Tensor *x = tensor_rand_normal(s, xsh, 2, 1.0f, &rng);

    ClsArg a;
    memset(&a, 0, sizeof(a));
    a.x = x;
    a.labels = NULL;
    a.lab[0] = 0; a.lab[1] = 1; a.lab[2] = 2; a.lab[3] = 3; a.lab[4] = 0; a.lab[5] = 1;

    Sequential *mlp = sequential_create();
    sequential_add(mlp, (Module *)linear_create(s, 3, 5, 1));
    sequential_add(mlp, gelu_create());
    sequential_add(mlp, (Module *)linear_create(s, 5, 4, 1));
    fd_check_model("MLP (GELU) x-entropy", &mlp->base, mlp_loss, &a, 3e-2, 12);
    module_free(&mlp->base);

    Sequential *mlp2 = sequential_create();
    sequential_add(mlp2, (Module *)linear_create(s, 3, 4, 1));
    sequential_add(mlp2, relu_create());
    sequential_add(mlp2, (Module *)linear_create(s, 4, 4, 1));
    fd_check_model("MLP (ReLU) SSE", &mlp2->base, sse_loss, &a, 3e-2, 12);
    module_free(&mlp2->base);

    scope_destroy(s);
}

static void test_cnn(void) {
    MiniTorchScope *s = scope_create();
    unsigned long long rng = 7;
    int64_t xsh[4] = { 2, 1, 6, 6 };
    Tensor *x = tensor_rand_normal(s, xsh, 4, 1.0f, &rng);
    int64_t lsh[1] = { 2 };
    Tensor *labels = tensor_zeros(s, lsh, 1);
    labels->data[0] = 0.0f;
    labels->data[1] = 1.0f;
    ClsArg a;
    memset(&a, 0, sizeof(a));
    a.x = x;
    a.labels = labels;
    a.lab[0] = 0; a.lab[1] = 1;

    Sequential *cnn = sequential_create();
    sequential_add(cnn, (Module *)conv2d_create(s, 1, 2, 3, 1, 1, 1, 1));
    sequential_add(cnn, relu_create());
    sequential_add(cnn, (Module *)maxpool2d_create(2, 2));
    sequential_add(cnn, flatten_create());
    sequential_add(cnn, (Module *)linear_create(s, 2 * 3 * 3, 2, 1));
    fd_check_model("CNN conv+pool+linear x-entropy", &cnn->base, mlp_loss, &a, 4e-2, 12);
    module_free(&cnn->base);
    scope_destroy(s);
}

static void test_layernorm(void) {
    MiniTorchScope *s = scope_create();
    unsigned long long rng = 31;
    int64_t xsh[2] = { 4, 6 };
    Tensor *x = tensor_rand_normal(s, xsh, 2, 1.0f, &rng);
    ClsArg a;
    memset(&a, 0, sizeof(a));
    a.x = x;
    a.labels = NULL;
    a.lab[0] = 1; a.lab[1] = 0; a.lab[2] = 1; a.lab[3] = 0;

    Sequential *ln = sequential_create();
    sequential_add(ln, (Module *)layernorm1d_create(s, 6));
    sequential_add(ln, (Module *)linear_create(s, 6, 3, 1));
    fd_check_model("LayerNorm + Linear SSE", &ln->base, sse_loss, &a, 3e-2, 12);
    module_free(&ln->base);
    scope_destroy(s);
}

static void test_transformer(void) {
    MiniTorchScope *s = scope_create();
    unsigned long long rng = 5;
    int64_t xsh[2] = { 4, 8 };             /* [T, dim] */
    Tensor *x = tensor_rand_normal(s, xsh, 2, 0.5f, &rng);
    ClsArg a;
    memset(&a, 0, sizeof(a));
    a.x = x;
    a.labels = NULL;
    a.lab[0] = 1; a.lab[1] = 0; a.lab[2] = 1; a.lab[3] = 0;

    TransformerBlock *blk = transformer_block_create(s, 8, 2, 8, &rng);
    fd_check_model("TransformerBlock SSE", &blk->base, sse_loss, &a, 8e-2, 16);
    module_free(&blk->base);
    scope_destroy(s);
}

/* ---- optimizer sanity -------------------------------------------- */

static void test_optimizer_descends(void) {
    MiniTorchScope *s = scope_create();
    unsigned long long rng = 3;
    int64_t sh[2] = { 8, 2 };
    Tensor *x = tensor_rand_normal(s, sh, 2, 1.0f, &rng);

    Sequential *mlp = sequential_create();
    sequential_add(mlp, (Module *)linear_create(s, 2, 8, 1));
    sequential_add(mlp, gelu_create());
    sequential_add(mlp, (Module *)linear_create(s, 8, 1, 1));

    int nparams = 0;
    Tensor **params = module_params(&mlp->base, &nparams);
    Optimizer *opt = optim_adam(params, nparams, 0.05f, 0.9f, 0.999f, 1e-8f);

    /* target: y = x0 - x1 (a linear function the net must approximate) */
    Tensor *target = tensor_zeros(s, (int64_t[]){ 8, 1 }, 2);
    for (int i = 0; i < 8; i++) target->data[i] = x->data[i * 2] - x->data[i * 2 + 1];

    float first = 0.0f, last = 0.0f;
    for (int it = 0; it < 400; it++) {
        optim_zero_grad(opt);
        Tensor *out = module_apply(&mlp->base, x);
        Tensor *diff = tensor_sub(out, target);
        Tensor *loss = tensor_sum(tensor_mul(diff, diff));
        if (it == 0) first = loss->data[0];
        last = loss->data[0];
        tensor_backward(loss, NULL);
        optim_step(opt);
        tensor_free(loss);
    }
    printf("  [%s] %-34s loss %.4f -> %.6f\n", last < first * 0.05f ? " ok " : "FAIL",
           "Adam reduces regression loss", first, last);
    if (last < first * 0.05f) g_pass++; else g_fail++;

    optim_free(opt);
    free(params);
    module_free(&mlp->base);
    scope_destroy(s);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    mt_seed(4242);
    printf("miniTorch module gradient checks\n");
    printf("----------------------------------\n");
    test_mlp();
    test_cnn();
    test_layernorm();
    test_transformer();
    test_optimizer_descends();
    printf("----------------------------------\n");
    printf("passed %d, failed %d\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
