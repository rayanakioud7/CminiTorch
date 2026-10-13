/* test_autograd.c -- finite-difference gradient checks for the engine.
 *
 * Every check compares an analytic gradient produced by tensor_backward()
 * against a central difference computed by tensor_numeric_grad().
 */
#include "tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int g_fail = 0;
static int g_pass = 0;

static void report(const char *name, float worst, float tol) {
    if (worst > tol) {
        printf("  [FAIL] %-30s worst rel err %.3e (tol %.0e)\n", name, worst, tol);
        g_fail++;
    } else {
        printf("  [ ok ] %-30s worst rel err %.3e\n", name, worst);
        g_pass++;
    }
}

static float rel_err(float analytic, float numeric) {
    float denom = fabsf(analytic) > 1e-3f ? fabsf(analytic) : 1.0f;
    return fabsf(analytic - numeric) / denom;
}

/* ---- fixtures ---------------------------------------------------- */

typedef struct { Tensor *other; } TwoArg;

static Tensor *f_mul(Tensor *p, void *ud) {
    TwoArg *t = (TwoArg *)ud;
    return tensor_sum(tensor_mul(p, t->other));
}
static Tensor *f_div(Tensor *p, void *ud) {
    TwoArg *t = (TwoArg *)ud;
    return tensor_sum(tensor_div(p, t->other));
}
static Tensor *f_add_bcast(Tensor *p, void *ud) {
    TwoArg *t = (TwoArg *)ud;
    return tensor_sum(tensor_add(p, t->other)); /* [2,3] + [3] */
}
static Tensor *f_matmul(Tensor *p, void *ud) {
    TwoArg *t = (TwoArg *)ud;                /* p [2,3] @ other [3,4] */
    return tensor_sum(tensor_matmul(p, t->other));
}
static Tensor *f_matmul_tn(Tensor *p, void *ud) {
    TwoArg *t = (TwoArg *)ud;                /* p [2,3] @ other [4,3]^T */
    return tensor_sum(tensor_matmul_tn(p, t->other));
}
static Tensor *f_relu(Tensor *p, void *ud) { (void)ud; return tensor_sum(tensor_relu(p)); }
static Tensor *f_sigmoid(Tensor *p, void *ud) { (void)ud; return tensor_sum(tensor_sigmoid(p)); }
static Tensor *f_tanh(Tensor *p, void *ud) { (void)ud; return tensor_sum(tensor_tanh(p)); }
static Tensor *f_gelu(Tensor *p, void *ud) { (void)ud; return tensor_sum(tensor_gelu(p)); }
static Tensor *f_exp(Tensor *p, void *ud) { (void)ud; return tensor_sum(tensor_exp(p)); }
static Tensor *f_log(Tensor *p, void *ud) { (void)ud; return tensor_sum(tensor_log(p)); }
static Tensor *f_sqrt(Tensor *p, void *ud) { (void)ud; return tensor_sum(tensor_sqrt(p)); }
static Tensor *f_pow(Tensor *p, void *ud) { (void)ud; return tensor_sum(tensor_pow(p, 3.0f)); }
static Tensor *f_sumdim(Tensor *p, void *ud) { (void)ud; return tensor_sum(tensor_sum_dim(p, 1, 1)); }
static Tensor *f_meandim(Tensor *p, void *ud) { (void)ud; return tensor_sum(tensor_mean_dim(p, 1, 1)); }

typedef struct { Tensor *targets; } CEArg;

static Tensor *f_cross_entropy(Tensor *p, void *ud) {
    CEArg *a = (CEArg *)ud;
    return tensor_cross_entropy(p, a->targets, 1);
}

/* ---- generic unary-by-finite-difference -------------------------- */

static void check_unary(const char *name, Tensor *(*f)(Tensor *, void *), float tol) {
    MiniTorchScope *s = scope_create();
    int64_t sh[2] = { 3, 4 };
    Tensor *p = tensor_create(s, sh, 2, NULL, 1);
    for (int64_t i = 0; i < p->numel; i++)
        p->data[i] = ((float)(i % 7) - 3.0f) * 0.4f + 0.15f; /* mixed signs */

    Tensor *loss = f(p, NULL);
    tensor_backward(loss, NULL);

    float worst = 0.0f;
    for (int64_t i = 0; i < p->numel; i++) {
        int64_t idx[2] = { i / 4, i % 4 };
        float num = tensor_numeric_grad(p, idx, 1e-3f, f, NULL);
        float rel = rel_err(p->grad[i], num);
        if (rel > worst) worst = rel;
    }
    report(name, worst, tol);
    tensor_free(loss);
    scope_destroy(s);
}

/* ---- binary ops --------------------------------------------------- */

static void check_binary(const char *name, Tensor *(*f)(Tensor *, void *), float tol, int bcast) {
    MiniTorchScope *s = scope_create();
    unsigned long long rng = 20240607;
    int64_t psh[2] = { 2, 3 };
    int64_t osh[2] = { 2, 3 };
    int odeg = 2;
    if (bcast) { osh[0] = 1; osh[1] = 3; }

    Tensor *p = tensor_rand_normal(s, psh, 2, 0.8f, &rng);
    p->requires_grad = 1;
    Tensor *other = tensor_rand_normal(s, osh, odeg, 0.8f, &rng);
    for (int64_t i = 0; i < other->numel; i++) other->data[i] += 1.6f; /* keep div well posed */

    TwoArg arg = { other };
    Tensor *loss = f(p, &arg);
    tensor_backward(loss, NULL);

    float worst = 0.0f;
    for (int64_t i = 0; i < p->numel; i++) {
        int64_t idx[2] = { i / 3, i % 3 };
        float num = tensor_numeric_grad(p, idx, 1e-3f, f, &arg);
        float rel = rel_err(p->grad[i], num);
        if (rel > worst) worst = rel;
    }
    report(name, worst, tol);
    tensor_free(loss);
    scope_destroy(s);
}

/* ---- matmul (needs its own operand shapes) ------------------------ */

static void check_matmul(const char *name, Tensor *(*f)(Tensor *, void *), int tn, float tol) {
    MiniTorchScope *s = scope_create();
    unsigned long long rng = 777;
    int64_t psh[2] = { 2, 3 };
    int64_t osh[2];
    if (tn) { osh[0] = 4; osh[1] = 3; }   /* matmul_tn: [4,3] */
    else    { osh[0] = 3; osh[1] = 4; }   /* matmul:    [3,4] */

    Tensor *p = tensor_rand_normal(s, psh, 2, 0.8f, &rng);
    p->requires_grad = 1;
    Tensor *other = tensor_rand_normal(s, osh, 2, 0.8f, &rng);

    TwoArg arg = { other };
    Tensor *loss = f(p, &arg);
    tensor_backward(loss, NULL);

    float worst = 0.0f;
    for (int64_t i = 0; i < p->numel; i++) {
        int64_t idx[2] = { i / 3, i % 3 };
        float num = tensor_numeric_grad(p, idx, 1e-3f, f, &arg);
        float rel = rel_err(p->grad[i], num);
        if (rel > worst) worst = rel;
    }
    report(name, worst, tol);
    tensor_free(loss);
    scope_destroy(s);
}

/* ---- conv2d / maxpool / embedding -------------------------------- */

static float sum_of(Tensor *t) {
    float a = 0.0f;
    float *d = tensor_data(t);
    for (int64_t i = 0; i < tensor_numel(t); i++) a += d[i];
    return a;
}

static void check_conv(void) {
    MiniTorchScope *s = scope_create();
    unsigned long long rng = 12345;
    int64_t xs[4] = { 2, 3, 5, 5 };
    int64_t ws[4] = { 4, 3, 3, 3 };
    int64_t bs[1] = { 4 };
    Tensor *x = tensor_rand_normal(s, xs, 4, 0.6f, &rng);
    Tensor *w = tensor_rand_normal(s, ws, 4, 0.4f, &rng);
    w->requires_grad = 1;
    Tensor *b = tensor_rand_normal(s, bs, 1, 0.2f, &rng);
    b->requires_grad = 1;

    /* ONE analytic pass, at the unperturbed weights. */
    Tensor *loss = tensor_sum(tensor_relu(tensor_conv2d(x, w, b, 1, 1, 1)));
    tensor_backward(loss, NULL);
    tensor_free(loss);

    /* Then finite differences, which must not touch the analytic gradients. */
    float worst = 0.0f;
    for (int64_t k = 0; k < 8; k++) {
        int64_t off = (k * 13) % w->numel;
        float ana = w->grad[off];
        w->data[off] += 1e-3f;
        Tensor *yp = tensor_relu(tensor_conv2d(x, w, b, 1, 1, 1));
        float plus = sum_of(yp);
        tensor_free(yp);
        w->data[off] -= 2e-3f;
        Tensor *ym = tensor_relu(tensor_conv2d(x, w, b, 1, 1, 1));
        float minus = sum_of(ym);
        tensor_free(ym);
        w->data[off] += 1e-3f;
        float rel = rel_err(ana, (plus - minus) / 2e-3f);
        if (rel > worst) worst = rel;
    }
    for (int64_t k = 0; k < 4; k++) {
        float ana = b->grad[k];
        b->data[k] += 1e-3f;
        Tensor *yp = tensor_relu(tensor_conv2d(x, w, b, 1, 1, 1));
        float plus = sum_of(yp);
        tensor_free(yp);
        b->data[k] -= 2e-3f;
        Tensor *ym = tensor_relu(tensor_conv2d(x, w, b, 1, 1, 1));
        float minus = sum_of(ym);
        tensor_free(ym);
        b->data[k] += 1e-3f;
        float rel = rel_err(ana, (plus - minus) / 2e-3f);
        if (rel > worst) worst = rel;
    }
    report("conv2d weight+bias grad", worst, 3e-2f);
    scope_destroy(s);
}

static void check_pool_embed(void) {
    MiniTorchScope *s = scope_create();
    unsigned long long rng = 999;
    int64_t xs[4] = { 1, 2, 6, 6 };
    Tensor *x = tensor_rand_normal(s, xs, 4, 1.0f, &rng);
    x->requires_grad = 1;
    Tensor *p = tensor_maxpool2d(x, 2, 2);
    Tensor *loss = tensor_sum(p);
    tensor_backward(loss, NULL);

    float total = 0.0f;
    int picked = 0;
    for (int64_t i = 0; i < x->numel; i++) {
        total += x->grad[i];
        if (x->grad[i] > 0.5f) picked++;
    }
    int ok = fabsf(total - (float)p->numel) < 1e-4f && picked == (int)p->numel;
    printf("  [%s] %-30s routed=%d pooled=%lld sum=%.1f\n", ok ? " ok " : "FAIL",
           "maxpool2d grad routing", picked, (long long)p->numel, total);
    if (ok) g_pass++; else g_fail++;
    tensor_free(loss);

    /* embedding: gradient must scatter-add into the touched rows */
    int64_t wsh[2] = { 5, 3 };
    Tensor *w = tensor_rand_normal(s, wsh, 2, 0.5f, &rng);
    w->requires_grad = 1;
    int64_t idx[4] = { 2, 0, 2, 4 };
    Tensor *e = tensor_embedding(w, idx, 4);
    Tensor *eloss = tensor_sum(e);
    tensor_backward(eloss, NULL);
    ok = 1;
    for (int64_t r = 0; r < 5; r++) {
        float want = 0.0f;
        for (int i = 0; i < 4; i++) if (idx[i] == r) want += 1.0f;
        for (int64_t c = 0; c < 3; c++)
            if (fabsf(w->grad[r * 3 + c] - want) > 1e-5f) ok = 0;
    }
    printf("  [%s] %-30s hits from idx (2,0,2,4)\n", ok ? " ok " : "FAIL",
           "embedding scatter-add");
    if (ok) g_pass++; else g_fail++;
    tensor_free(eloss);
    scope_destroy(s);
}

/* ---- parameter sharing / accumulation ---------------------------- */

static void check_shared_param(void) {
    MiniTorchScope *s = scope_create();
    int64_t sh[1] = { 4 };
    Tensor *p = tensor_create(s, sh, 1, NULL, 1);
    for (int64_t i = 0; i < 4; i++) p->data[i] = 0.5f;

    Tensor *twice = tensor_add(p, p); /* both edges are the same tensor */
    Tensor *loss = tensor_sum(tensor_mul(twice, twice));
    tensor_backward(loss, NULL);
    float want = 8.0f * 0.5f; /* d/dp (2p)^2 = 8p */
    int ok = fabsf(p->grad[0] - want) < 1e-4f;
    printf("  [%s] %-30s got %.4f want %.4f\n", ok ? " ok " : "FAIL",
           "same operand on both edges", p->grad[0], want);
    if (ok) g_pass++; else g_fail++;
    tensor_free(loss);
    scope_destroy(s);
}

static void check_two_consumers(void) {
    MiniTorchScope *s = scope_create();
    unsigned long long rng = 7;
    int64_t sh[2] = { 3, 3 };
    Tensor *p = tensor_rand_normal(s, sh, 2, 0.7f, &rng);
    p->requires_grad = 1;

    Tensor *la = tensor_sum(tensor_mul(p, p)); /* first pass: 2p */
    tensor_backward(la, NULL);
    int ok = 1;
    for (int i = 0; i < 9; i++)
        if (fabsf(p->grad[i] - 2.0f * p->data[i]) > 1e-4f) ok = 0;
    tensor_free(la);

    Tensor *lb = tensor_sum(tensor_add(tensor_mul(p, p), tensor_mul_scalar(p, 3.0f)));
    tensor_backward(lb, NULL); /* must ADD to the existing gradient */
    for (int i = 0; i < 9; i++)
        if (fabsf(p->grad[i] - (4.0f * p->data[i] + 3.0f)) > 1e-4f) ok = 0;
    printf("  [%s] %-30s\n", ok ? " ok " : "FAIL", "two backward passes accumulate");
    if (ok) g_pass++; else g_fail++;
    tensor_free(lb);
    scope_destroy(s);
}

/* ---- losses ------------------------------------------------------ */

static void check_cross_entropy(void) {
    MiniTorchScope *s = scope_create();
    unsigned long long rng = 4242;
    int64_t sh[2] = { 4, 5 };
    Tensor *p = tensor_rand_normal(s, sh, 2, 1.0f, &rng);
    for (int64_t i = 0; i < p->numel; i++) p->data[i] += 0.3f * (float)(i % 5);
    p->requires_grad = 1;
    Tensor *targets = tensor_zeros(s, sh, 2);
    for (int i = 0; i < 4; i++) targets->data[i * 5 + (i % 5)] = 1.0f;

    Tensor *loss = tensor_cross_entropy(p, targets, 1);
    tensor_backward(loss, NULL);

    CEArg arg = { targets };
    float worst = 0.0f;
    for (int64_t i = 0; i < p->numel; i++) {
        int64_t idx[2] = { i / 5, i % 5 };
        float num = tensor_numeric_grad(p, idx, 1e-3f, f_cross_entropy, &arg);
        float rel = rel_err(p->grad[i], num);
        if (rel > worst) {
            worst = rel;
            printf("       [ce] i=%lld ana=%.6f num=%.6f\n", (long long)i, p->grad[i], num);
        }
    }
    report("cross_entropy grad", worst, 2e-2f);
    tensor_free(loss);
    scope_destroy(s);
}

/* ---- leak check --------------------------------------------------- */

static void check_leaks(void) {
    MiniTorchScope *s = scope_create();
    unsigned long long rng = 5;
    int64_t sh[2] = { 4, 4 };
    Tensor *p = tensor_rand_normal(s, sh, 2, 0.5f, &rng);
    p->requires_grad = 1;
    size_t before = scope_live_tensors(s);
    for (int it = 0; it < 25; it++) {
        Tensor *h = tensor_relu(tensor_matmul(p, p));
        Tensor *loss = tensor_sum(tensor_gelu(h));
        tensor_backward(loss, NULL);
        tensor_free(loss);
    }
    size_t after = scope_live_tensors(s);
    if (after != before) {
        printf("       leaked %zu tensors:\n", after - before);
        int shown = 0;
        for (Tensor *t = scope_first(s); t && shown < 8; t = tensor_scope_next(t), shown++)
            printf("         op=%-10s refs=%d numel=%lld\n", tensor_op_name(t), tensor_refs(t),
                   (long long)tensor_numel(t));
    }
    printf("  [%s] %-30s live before=%zu after=%zu\n", after == before ? " ok " : "FAIL",
           "25 iterations leave no leak", before, after);
    if (after == before) g_pass++; else g_fail++;
    scope_destroy(s);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("miniTorch autograd gradient checks\n");
    printf("----------------------------------\n");

    check_unary("relu", f_relu, 2e-2f);
    check_unary("sigmoid", f_sigmoid, 2e-2f);
    check_unary("tanh", f_tanh, 2e-2f);
    check_unary("gelu", f_gelu, 2e-2f);
    check_unary("exp", f_exp, 2e-2f);
    check_unary("log", f_log, 2e-2f);
    check_unary("sqrt", f_sqrt, 2e-2f);
    check_unary("pow(3)", f_pow, 2e-2f);
    check_unary("sum_dim(1,keepdim)", f_sumdim, 2e-2f);
    check_unary("mean_dim(1,keepdim)", f_meandim, 2e-2f);

    check_binary("mul (same shape)", f_mul, 2e-2f, 0);
    check_binary("div (same shape)", f_div, 2e-2f, 0);
    check_binary("mul (broadcast [3])", f_mul, 2e-2f, 1);
    check_binary("add (broadcast [3])", f_add_bcast, 2e-2f, 1);

    check_matmul("matmul", f_matmul, 0, 2e-2f);
    check_matmul("matmul_tn", f_matmul_tn, 1, 2e-2f);

    check_conv();
    check_pool_embed();
    check_shared_param();
    check_two_consumers();
    check_cross_entropy();
    check_leaks();

    printf("----------------------------------\n");
    printf("passed %d, failed %d\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
