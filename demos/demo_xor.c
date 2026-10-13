/* demo_xor.c -- a tiny MLP learning XOR.
 *
 * XOR is the smallest problem a linear model cannot fit, so it exercises the
 * whole stack: modules, autograd, optimizer, loss.
 *
 * The demo deliberately runs the same architecture twice, once with ReLU and
 * once with GELU, and prints the final decision per input. With a bias-free
 * narrow ReLU hidden layer and a large step, one unit tends to die and the
 * model settles on "always predict 0" (3/4 rather than 4/4) -- exactly the
 * local minimum you would hit in PyTorch with the same setup.
 */
#include "data.h"
#include "nn.h"
#include "optim.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

typedef enum { ACT_RELU, ACT_GELU } ActKind;

static int run_trial(Dataset *ds, ActKind act, int use_adam, int epochs, int verbose, int clip) {
    MiniTorchScope *s = ds->inputs->scope;

    Sequential *net = sequential_create();
    sequential_add(net, (Module *)linear_create(s, 2, 8, 1));
    sequential_add(net, act == ACT_RELU ? relu_create() : gelu_create());
    sequential_add(net, (Module *)linear_create(s, 8, 2, 1));

    int nparams = 0;
    Tensor **params = module_params(&net->base, &nparams);
    Optimizer *opt = use_adam ? optim_adam(params, nparams, 0.05f, 0.9f, 0.999f, 1e-8f)
                              : optim_sgd(params, nparams, 0.02f, 0.9f, 0.0f);

    int64_t labels[4];
    for (int i = 0; i < 4; i++) labels[i] = (int64_t)ds->targets->data[i];

    if (verbose) {
        printf("\n== XOR  act=%-4s  opt=%-4s ==\n", act == ACT_RELU ? "ReLU" : "GELU",
               use_adam ? "Adam" : "SGD");
        module_print(&net->base, 0);
    }

    float last = 0.0f;
    for (int epoch = 0; epoch < epochs; epoch++) {
        optim_zero_grad(opt);
        Tensor *out = module_apply(&net->base, ds->inputs);
        Tensor *loss = loss_cross_entropy_labels(out, labels, 4);
        last = loss->data[0];
        tensor_backward(loss, NULL);

        /* Clip the global gradient norm. Without this the ReLU trial can take
         * a very large step on a confidently-wrong example and diverge. */
        double sq = 0.0;
        for (int i = 0; i < nparams; i++)
            for (int64_t k = 0; k < params[i]->numel; k++) {
                float g = params[i]->grad[k];
                sq += (double)g * g;
            }
        double norm = sqrt(sq);
        if (norm > 5.0 && clip) {
            float scale = (float)(5.0 / norm);
            for (int i = 0; i < nparams; i++)
                for (int64_t k = 0; k < params[i]->numel; k++) params[i]->grad[k] *= scale;
        }

        optim_step(opt);
        if (verbose && (epoch % (epochs / 5) == 0 || epoch == epochs - 1))
            printf("  epoch %5d  loss %.5f\n", epoch, last);
        tensor_free(loss);
    }

    Tensor *out = module_apply(&net->base, ds->inputs);
    int correct = 0;
    for (int i = 0; i < 4; i++) {
        float a = out->data[i * 2 + 0], b = out->data[i * 2 + 1];
        int pred = a > b ? 0 : 1;
        if (pred == labels[i]) correct++;
        if (verbose)
            printf("    x=(%.0f,%.0f)  logits=(%+7.3f,%+7.3f)  pred=%d  target=%lld  %s\n",
                   ds->inputs->data[i * 2], ds->inputs->data[i * 2 + 1], a, b, pred,
                   (long long)labels[i], pred == labels[i] ? "ok" : "WRONG");
    }
    printf("  %-4s + %-4s : final loss %8.5f   accuracy %d/4\n", act == ACT_RELU ? "ReLU" : "GELU",
           use_adam ? "Adam" : "SGD", last, correct);

    optim_free(opt);
    free(params);
    module_free(&net->base);
    return correct;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    mt_seed(1234);

    MiniTorchScope *s = scope_create();
    Dataset *ds = dataset_xor(s);

    printf("XOR: 4 points, 2 classes. A linear model cannot fit this.\n");

    /* One verbose run so the training curve is visible. */
    run_trial(ds, ACT_GELU, 1, 3000, 1, 1);

    printf("\nsummary (3000 epochs each)\n");
    run_trial(ds, ACT_RELU, 0, 3000, 0, 1);
    run_trial(ds, ACT_RELU, 1, 3000, 0, 1);
    run_trial(ds, ACT_GELU, 0, 3000, 0, 1);
    run_trial(ds, ACT_GELU, 1, 3000, 0, 1);

    dataset_free(ds);
    scope_destroy(s);
    printf("\ndone.\n");
    return 0;
}
