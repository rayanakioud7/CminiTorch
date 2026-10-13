/* demo_transformer.c -- A generative causal Transformer language model in C.
 *
 * Implements a nanoGPT-style autoregressive Transformer that learns a
 * character-level language model over repeating text motifs.
 *
 * Architecture:
 *   Token IDs [T]
 *   -> Embedding(vocab_size -> d_model) -> [T, d_model]
 *   + Learned Positional Embedding [T, d_model]
 *   -> TransformerBlock(d_model, n_heads, block_size)
 *      - Pre-LayerNorm 1
 *      - Causal Multi-Head Self-Attention + Residual
 *      - Pre-LayerNorm 2
 *      - MLP (Linear -> GELU -> Linear) + Residual
 *   -> LM Head: Linear(d_model -> vocab_size)
 *   -> Cross-Entropy Loss over sequence tokens
 *   -> Autoregressive Text Generation
 */
#include "data.h"
#include "nn.h"
#include "optim.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define VOCAB_SIZE 12
#define D_MODEL    16
#define N_HEADS    2
#define BLOCK_SIZE 8

typedef struct {
    Embedding        *tok_emb;
    Tensor           *pos_emb;   /* [BLOCK_SIZE, D_MODEL] */
    TransformerBlock *block;
    Linear           *lm_head;
    MiniTorchScope   *scope;
} NanoGPT;

static NanoGPT *nanogpt_create(MiniTorchScope *s) {
    NanoGPT *model = (NanoGPT *)calloc(1, sizeof(NanoGPT));
    model->scope = s;
    model->tok_emb = embedding_create(s, VOCAB_SIZE, D_MODEL);

    int64_t pos_sh[2] = { BLOCK_SIZE, D_MODEL };
    model->pos_emb = tensor_rand_normal(s, pos_sh, 2, 0.02f, NULL);
    model->pos_emb->requires_grad = 1;
    model->pos_emb->held = 1;

    model->block = transformer_block_create(s, D_MODEL, N_HEADS, BLOCK_SIZE, NULL);
    model->lm_head = linear_create(s, D_MODEL, VOCAB_SIZE, 0);
    return model;
}

static void nanogpt_free(NanoGPT *m) {
    if (!m) return;
    module_free(&m->tok_emb->base);
    module_free(&m->block->base);
    module_free(&m->lm_head->base);
    free(m);
}

static int collect_gpt_params(NanoGPT *m, Tensor ***out) {
    int cap = 32, n = 0;
    *out = (Tensor **)malloc(sizeof(Tensor *) * (size_t)cap);

    int count = 0;
    Tensor **p_emb = module_params(&m->tok_emb->base, &count);
    for (int i = 0; i < count; i++) (*out)[n++] = p_emb[i];
    free(p_emb);

    (*out)[n++] = m->pos_emb;

    Tensor **p_blk = module_params(&m->block->base, &count);
    for (int i = 0; i < count; i++) (*out)[n++] = p_blk[i];
    free(p_blk);

    Tensor **p_head = module_params(&m->lm_head->base, &count);
    for (int i = 0; i < count; i++) (*out)[n++] = p_head[i];
    free(p_head);

    return n;
}

static Tensor *nanogpt_forward(NanoGPT *m, const int64_t *tok_ids, int64_t T) {
    /* 1. Token embeddings: [T, D_MODEL] */
    Tensor *tok = embedding_forward(m->tok_emb, tok_ids, T);

    /* 2. Position embeddings slice: [T, D_MODEL] */
    int64_t sh[2] = { T, D_MODEL };
    Tensor *pos = tensor_zeros(m->scope, sh, 2);
    for (int64_t i = 0; i < T; i++) {
        for (int64_t d = 0; d < D_MODEL; d++) {
            pos->data[i * D_MODEL + d] = m->pos_emb->data[i * D_MODEL + d];
        }
    }

    Tensor *x = tensor_add(tok, pos);

    /* 3. Transformer Block: [T, D_MODEL] */
    Tensor *h = module_forward(&m->block->base, x);

    /* 4. LM Head: [T, VOCAB_SIZE] */
    Tensor *logits = module_forward(&m->lm_head->base, h);
    return logits;
}

static void sample_sequence(NanoGPT *m, const char *prompt, int steps) {
    int len = (int)strlen(prompt);
    int64_t current[BLOCK_SIZE];
    memset(current, 0, sizeof(current));

    char output[128];
    snprintf(output, sizeof(output), "%s", prompt);
    int cur_len = len;

    printf("  Prompt:    \"%s\"\n", prompt);
    printf("  Generated: \"%s", prompt);

    for (int step = 0; step < steps; step++) {
        int ctx_len = cur_len < BLOCK_SIZE ? cur_len : BLOCK_SIZE;
        int start = cur_len - ctx_len;
        for (int i = 0; i < ctx_len; i++) {
            current[i] = (int64_t)(output[start + i] - 'a');
            if (current[i] < 0 || current[i] >= VOCAB_SIZE) current[i] = 0;
        }

        Tensor *logits = nanogpt_forward(m, current, ctx_len);
        /* Pick argmax from last token's logits */
        float *last_logits = logits->data + (ctx_len - 1) * VOCAB_SIZE;
        int best = 0;
        float best_val = last_logits[0];
        for (int v = 1; v < VOCAB_SIZE; v++) {
            if (last_logits[v] > best_val) {
                best_val = last_logits[v];
                best = v;
            }
        }
        char next_ch = (char)('a' + best);
        printf("%c", next_ch);
        if (cur_len < (int)sizeof(output) - 2) {
            output[cur_len++] = next_ch;
            output[cur_len] = '\0';
        }
        tensor_free(logits);
    }
    printf("\"\n");
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    mt_seed(1337);

    printf("============================================================\n");
    printf(" miniTorch Transformer Demo: Character-Level Language Model\n");
    printf("============================================================\n");

    MiniTorchScope *s = scope_create();
    /* A compact corpus containing repetitive motifs "abcabc", "deffde", "ghigih" */
    char corpus[256];
    corpus[0] = '\0';
    for (int rep = 0; rep < 10; rep++) {
        strcat(corpus, "abcabcdeffdeghigih");
    }
    printf("\nCorpus preview (length %zu chars):\n  \"%s\"\n", strlen(corpus), corpus);

    Dataset *ds = dataset_text(s, corpus, BLOCK_SIZE);
    printf("\nTraining Dataset: %lld sequence pairs (context window = %d)\n",
           (long long)ds->n, BLOCK_SIZE);

    NanoGPT *gpt = nanogpt_create(s);

    Tensor **params = NULL;
    int nparams = collect_gpt_params(gpt, &params);
    int64_t total_weights = 0;
    for (int i = 0; i < nparams; i++) total_weights += params[i]->numel;
    printf("\nModel Parameters:\n");
    printf("  Vocab: %d | Embedding Dim: %d | Heads: %d | Context: %d\n",
           VOCAB_SIZE, D_MODEL, N_HEADS, BLOCK_SIZE);
    printf("  Total learnable weights: %lld floats across %d tensors\n",
           (long long)total_weights, nparams);

    Optimizer *opt = optim_adam(params, nparams, 0.005f, 0.9f, 0.999f, 1e-8f);

    printf("\nTraining (Adam, lr=0.005, 50 epochs):\n");
    int64_t in_ids[BLOCK_SIZE], tgt_ids[BLOCK_SIZE];

    for (int epoch = 0; epoch <= 50; epoch++) {
        float epoch_loss = 0.0f;
        int num_batches = (int)ds->n;

        for (int sample = 0; sample < num_batches; sample++) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                in_ids[j]  = (int64_t)ds->inputs->data[sample * BLOCK_SIZE + j];
                tgt_ids[j] = (int64_t)ds->targets->data[sample * BLOCK_SIZE + j];
            }

            optim_zero_grad(opt);
            Tensor *logits = nanogpt_forward(gpt, in_ids, BLOCK_SIZE);
            Tensor *loss = loss_cross_entropy_labels(logits, tgt_ids, BLOCK_SIZE);
            epoch_loss += loss->data[0];

            if (epoch < 50) {
                tensor_backward(loss, NULL);

                /* Gradient clipping: clip global norm to 1.0 */
                double sq = 0.0;
                for (int i = 0; i < nparams; i++)
                    for (int64_t k = 0; k < params[i]->numel; k++) {
                        float g = params[i]->grad[k];
                        sq += (double)g * g;
                    }
                double norm = sqrt(sq);
                if (norm > 1.0) {
                    float scale = (float)(1.0 / norm);
                    for (int i = 0; i < nparams; i++)
                        for (int64_t k = 0; k < params[i]->numel; k++) params[i]->grad[k] *= scale;
                }

                optim_step(opt);
            }
            tensor_free(loss);
        }

        epoch_loss /= (float)num_batches;
        if (epoch % 10 == 0 || epoch == 50) {
            printf("  Epoch %3d | Mean Sequence Cross-Entropy Loss: %.5f\n", epoch, epoch_loss);
        }
    }

    printf("\nAutoregressive Text Generation (Motif Completion):\n");
    sample_sequence(gpt, "abc", 15);
    sample_sequence(gpt, "def", 15);
    sample_sequence(gpt, "ghi", 15);

    optim_free(opt);
    free(params);
    nanogpt_free(gpt);
    dataset_free(ds);
    scope_destroy(s);

    printf("\nTransformer Demo successfully completed!\n");
    return 0;
}
