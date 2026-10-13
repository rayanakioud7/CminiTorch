/* demo_cnn.c -- A convolutional neural network learning image classification.
 *
 * Trains a ConvNet on the synthetic "bars" dataset: 8x8 images where class 0
 * contains a vertical bar and class 1 contains a horizontal bar, corrupted
 * with random pixel noise.
 *
 * Architecture:
 *   Input [N, 1, 8, 8]
 *   -> Conv2d(1 -> 4, 3x3 kernel, pad 1, stride 1)
 *   -> ReLU
 *   -> MaxPool2d(2x2, stride 2) -> [N, 4, 4, 4]
 *   -> Flatten -> [N, 64]
 *   -> Linear(64 -> 16)
 *   -> ReLU
 *   -> Linear(16 -> 2) -> logits [N, 2]
 */
#include "data.h"
#include "nn.h"
#include "optim.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void print_ascii_bar(const float *img, int size, int label, int pred) {
    printf("  Sample [true=%d, pred=%d]:\n", label, pred);
    for (int r = 0; r < size; r++) {
        printf("    ");
        for (int c = 0; c < size; c++) {
            float val = img[r * size + c];
            printf("%s", val > 0.5f ? "##" : "..");
        }
        printf("\n");
    }
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    mt_seed(42);

    printf("============================================================\n");
    printf(" miniTorch CNN Demo: Horizontal vs Vertical Bar Recognition\n");
    printf("============================================================\n");

    MiniTorchScope *s = scope_create();
    int size = 8;
    int n_train = 80; /* 40 per class */
    int n_test = 20;  /* 10 per class */

    Dataset *train_ds = dataset_bars(s, n_train / 2, size);
    Dataset *test_ds  = dataset_bars(s, n_test / 2, size);

    printf("\nDatasets generated:\n");
    printf("  Train: %lld samples (%d vertical, %d horizontal)\n",
           (long long)train_ds->n, n_train / 2, n_train / 2);
    printf("  Test:  %lld samples (%d vertical, %d horizontal)\n",
           (long long)test_ds->n, n_test / 2, n_test / 2);

    /* Build CNN */
    Sequential *cnn = sequential_create();
    sequential_add(cnn, (Module *)conv2d_create(s, 1, 4, 3, 1, 1, 1, 1));
    sequential_add(cnn, relu_create());
    sequential_add(cnn, (Module *)maxpool2d_create(2, 2));
    sequential_add(cnn, flatten_create());
    sequential_add(cnn, (Module *)linear_create(s, 4 * 4 * 4, 16, 1));
    sequential_add(cnn, relu_create());
    sequential_add(cnn, (Module *)linear_create(s, 16, 2, 1));

    printf("\nModel Architecture:\n");
    module_print(&cnn->base, 0);

    int nparams = 0;
    Tensor **params = module_params(&cnn->base, &nparams);
    printf("Total learnable parameters: %d\n", module_num_params(&cnn->base));

    Optimizer *opt = optim_adam(params, nparams, 0.02f, 0.9f, 0.999f, 1e-8f);

    int64_t train_labels[80];
    for (int i = 0; i < n_train; i++) train_labels[i] = (int64_t)train_ds->targets->data[i];

    int64_t test_labels[20];
    for (int i = 0; i < n_test; i++) test_labels[i] = (int64_t)test_ds->targets->data[i];

    printf("\nTraining (Adam, lr=0.02, 60 epochs):\n");
    for (int epoch = 0; epoch <= 60; epoch++) {
        optim_zero_grad(opt);
        Tensor *out = module_apply(&cnn->base, train_ds->inputs);
        Tensor *loss = loss_cross_entropy_labels(out, train_labels, n_train);
        float loss_val = loss->data[0];

        int train_correct = 0;
        for (int i = 0; i < n_train; i++) {
            int pred = out->data[i * 2 + 0] > out->data[i * 2 + 1] ? 0 : 1;
            if (pred == train_labels[i]) train_correct++;
        }

        if (epoch % 10 == 0 || epoch == 60) {
            /* Compute test accuracy */
            Tensor *test_out = module_apply(&cnn->base, test_ds->inputs);
            int test_correct = 0;
            for (int i = 0; i < n_test; i++) {
                int pred = test_out->data[i * 2 + 0] > test_out->data[i * 2 + 1] ? 0 : 1;
                if (pred == test_labels[i]) test_correct++;
            }
            printf("  Epoch %3d | Loss: %8.5f | Train Acc: %2d/%d (%.1f%%) | Test Acc: %2d/%d (%.1f%%)\n",
                   epoch, loss_val, train_correct, n_train, 100.0f * (float)train_correct / (float)n_train,
                   test_correct, n_test, 100.0f * (float)test_correct / (float)n_test);
        }

        if (epoch < 60) {
            tensor_backward(loss, NULL);
            optim_step(opt);
        }
        tensor_free(loss);
    }

    printf("\nSample Test Set Predictions:\n");
    Tensor *final_test_out = module_apply(&cnn->base, test_ds->inputs);
    print_ascii_bar(test_ds->inputs->data, size, (int)test_labels[0],
                    final_test_out->data[0] > final_test_out->data[1] ? 0 : 1);
    print_ascii_bar(test_ds->inputs->data + 10 * size * size, size, (int)test_labels[10],
                    final_test_out->data[20] > final_test_out->data[21] ? 0 : 1);

    optim_free(opt);
    free(params);
    module_free(&cnn->base);
    dataset_free(train_ds);
    dataset_free(test_ds);
    scope_destroy(s);

    printf("\nCNN Demo successfully completed!\n");
    return 0;
}
