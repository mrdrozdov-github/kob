#include <math.h>
#include <iostream>

#include "kob.h"
#include "batch_reader.h"
#include "gflags/gflags.h"

#define DO_EVAL false
#define PRINT_SAMPLE false

/*

TODO:

- [ ] batching
- [ ] don't need to specify data size

*/

DEFINE_string(train_data_file, "train_data.txt", "Data file");
DEFINE_string(train_labels_file, "train_labels.txt", "Data file");
DEFINE_string(eval_data_file, "test_data.txt", "Data file");
DEFINE_string(eval_labels_file, "test_labels.txt", "Data file");
DEFINE_string(weight1_file, "w1", "Data file");
DEFINE_string(weight2_file, "w2", "Data file");
DEFINE_int32(data_size, 100, "Data dim");
DEFINE_int32(eval_data_size, 100, "Data dim");
DEFINE_int32(batch_size, 32, "Data dim");
DEFINE_int32(inp_dim, 784, "Data dim");
DEFINE_int32(hidden_dim, 64, "Data dim");
DEFINE_int32(outp_dim, 10, "Data dim");
DEFINE_int32(steps, 1, "Data dim");
DEFINE_double(learning_rate, 0.001, "Data dim");

void print_mnist(float *item) {
    int size = 28;
    int i, j;
    for (j = 0; j < size; j++)
    {
        for (i = 0; i < size; i++)
        {
            std::cout << ceil(item[j*28 + i]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    string filename = "/Users/Andrew/Developer/kob/examples/h5mnist/train.h5";
    string datasetname = "images";
    int n = 55000;
    int size = 784;
    int batch_size = FLAGS_batch_size;
    BatchReader batch_reader = BatchReader(filename, datasetname, n, size);

    float batch[batch_size * size];

    // Print sample of data.
    if (PRINT_SAMPLE) {
        for (int i = 0; i < 3; i++) {
            batch_reader.read_item(batch + i * size, i);
            print_mnist(batch + i * size);
        }
    }

    return 0;

    gflags::ParseCommandLineFlags(&argc, &argv, true);
    gflags::ShutDownCommandLineFlags();

    THFile *train_data_file = THDiskFile_new(FLAGS_train_data_file.c_str(), "r", 0);
    THFile *train_labels_file = THDiskFile_new(FLAGS_train_labels_file.c_str(), "r", 0);
    THFile *eval_data_file = THDiskFile_new(FLAGS_eval_data_file.c_str(), "r", 0);
    THFile *eval_labels_file = THDiskFile_new(FLAGS_eval_labels_file.c_str(), "r", 0);
    THFile *weight1_file = THDiskFile_new(FLAGS_weight1_file.c_str(), "r", 0);
    THFile *weight2_file = THDiskFile_new(FLAGS_weight2_file.c_str(), "r", 0);

    int inp_dim = FLAGS_inp_dim;
    int hidden_dim = FLAGS_hidden_dim;
    int outp_dim = FLAGS_outp_dim;

    THFloatTensor *data = THFloatTensor_newWithSize2d(FLAGS_data_size, inp_dim);
    THLongTensor *labels = THLongTensor_newWithSize1d(FLAGS_data_size);

    THFloatTensor *eval_data = THFloatTensor_newWithSize2d(FLAGS_eval_data_size, inp_dim);
    THLongTensor *eval_labels = THLongTensor_newWithSize1d(FLAGS_eval_data_size);

    Linear *linear1 = new Linear(inp_dim, hidden_dim);
    Linear *linear2 = new Linear(hidden_dim, outp_dim);

    // Initialization
    printf("Reading data\n");
    readFloat(train_data_file, data);
    printf("Reading labels\n");
    readLong(train_labels_file, labels);
    printf("Reading data\n");
    readFloat(eval_data_file, eval_data);
    printf("Reading labels\n");
    readLong(eval_labels_file, eval_labels);
    printf("Reading linear1\n");
    readFloat(weight1_file, linear1->weight);
    printf("Reading linear2\n");
    readFloat(weight2_file, linear2->weight);

    THFile_free(train_data_file);
    THFile_free(train_labels_file);
    THFile_free(eval_data_file);
    THFile_free(eval_labels_file);
    THFile_free(weight1_file);
    THFile_free(weight2_file);

    // Fix labels
    THLongTensor_add(labels, labels, 1);
    THLongTensor_add(eval_labels, eval_labels, 1);

    // printf("weight1: %f\n", THFloatTensor_sumall(linear1->weight));
    // printf("weight1[0]: %f\n", THFloatTensor_data(linear1->weight)[0]);
    // printf("weight2: %f\n", THFloatTensor_sumall(linear2->weight));
    // printf("weight2[0]: %f\n", THFloatTensor_data(linear2->weight)[0]);

    Variable *train_var = new Variable(data);
    Variable *eval_var = new Variable(eval_data);
    THLongTensor *target = labels;
    THLongTensor *eval_target = eval_labels;
    Variable *inp_linear1;
    Variable *outp_linear1;
    Variable *inp_sigm;
    Variable *outp_sigm;
    Variable *inp_linear2;
    Variable *outp_linear2;
    Variable *inp_softmax;
    Variable *outp_softmax;
    Variable *inp_nll;
    Variable *outp_nll;

    Variable *grad_nll;
    Variable *grad_softmax;
    Variable *grad_linear2;
    Variable *grad_sigm;
    Variable *grad_linear1;

    for (int step = 0; step < FLAGS_steps; step++) {
        // TODO: Shuffling

        // Forward Pass
        // printf("forward (batch)(numel): %td\n", THFloatTensor_nElement(batch_var->data));

        inp_linear1 = train_var;
        outp_linear1 = linear1->forward(inp_linear1);
        // printf("forward (linear1)(numel): %td\n", THFloatTensor_nElement(outp_linear1->data));
        // printf("forward (linear1): %f\n", THFloatTensor_sumall(outp_linear1->data));

        inp_sigm = outp_linear1;
        outp_sigm = Sigmoid_forward(inp_sigm);
        // printf("forward (sigm)(numel): %td\n", THFloatTensor_nElement(outp_sigm->data));
        // printf("forward (sigm): %f\n", THFloatTensor_sumall(outp_sigm->data));

        inp_linear2 = outp_sigm;
        outp_linear2 = linear2->forward(inp_linear2);
        // printf("forward (linear2)(numel): %td\n", THFloatTensor_nElement(outp_linear2->data));
        // printf("forward (linear2): %f\n", THFloatTensor_sumall(outp_linear2->data));

        inp_softmax = outp_linear2;
        outp_softmax = LogSoftMax_forward(inp_softmax);
        // printf("forward (softmax)(numel): %td\n", THFloatTensor_nElement(outp_softmax->data));
        // printf("forward (softmax): %f\n", THFloatTensor_sumall(outp_softmax->data));

        inp_nll = outp_softmax;
        outp_nll = NLLLoss_forward(inp_nll, target);
        // printf("forward (nll)(numel): %td\n", THFloatTensor_nElement(outp_nll->data));
        printf("[%d] forward (nll): %f\n", step, THFloatTensor_sumall(outp_nll->data));

        // Backward Pass
        // THFloatTensor *loss = THFloatTensor_newWithSize2d(batch_size, outp_dim);
        // THFloatTensor_fill(loss, 1.0);
        // THFloatTensor_csub(loss, loss, 1.0, outp_linear2->data);
        // printf("loss: %f\n", THFloatTensor_sumall(loss));
        linear1->clear_grads();
        linear2->clear_grads();

        // printf("grads (linear1): %f\n", THFloatTensor_sumall(linear1->gradWeight));
        // printf("grads (linear2): %f\n", THFloatTensor_sumall(linear2->gradWeight));
        // printf("grads (linear1)[0]: %f\n", THFloatTensor_data(linear1->gradWeight)[0]);
        // printf("grads (linear2)[0]: %f\n", THFloatTensor_data(linear2->gradWeight)[0]);

        grad_nll = NLLLoss_backward(inp_nll, target);
        // printf("grad (nll): %f\n", THFloatTensor_sumall(grad_nll->data));
        // printf("grad (nll)(numel): %td\n", THFloatTensor_nElement(grad_nll->data));

        grad_softmax = LogSoftMax_backward(inp_softmax, outp_softmax->data, grad_nll->data);
        // printf("grad (softmax): %f\n", THFloatTensor_sumall(grad_softmax->data));
        // printf("grad (softmax)(numel): %td\n", THFloatTensor_nElement(grad_softmax->data));

        grad_linear2 = linear2->backward(inp_linear2, grad_softmax->data);
        // printf("grad (linear2): %f\n", THFloatTensor_sumall(grad_linear2->data));
        // printf("grad (linear2)(numel): %td\n", THFloatTensor_nElement(grad_linear2->data));

        grad_sigm = Sigmoid_backward(inp_sigm, outp_sigm->data, grad_linear2->data);
        // printf("grad (sigm): %f\n", THFloatTensor_sumall(grad_sigm->data));
        // printf("grad (sigm)(numel): %td\n", THFloatTensor_nElement(grad_sigm->data));

        grad_linear1 = linear1->backward(inp_linear1, grad_sigm->data);
        // printf("grad (linear1): %f\n", THFloatTensor_sumall(grad_linear1->data));
        // printf("grad (linear1)(numel): %td\n", THFloatTensor_nElement(grad_linear1->data));

        // printf("grads (linear1): %f\n", THFloatTensor_sumall(linear1->gradWeight));
        // printf("grads (linear1)[0]: %f\n", THFloatTensor_data(linear1->gradWeight)[0]);
        // printf("grads (linear2): %f\n", THFloatTensor_sumall(linear2->gradWeight));
        // printf("grads (linear2)[0]: %f\n", THFloatTensor_data(linear2->gradWeight)[0]);

        THFloatTensor_csub(linear1->weight, linear1->weight, FLAGS_learning_rate, linear1->gradWeight);
        THFloatTensor_csub(linear2->weight, linear2->weight, FLAGS_learning_rate, linear2->gradWeight);
        // printf("weight1: %f\n", THFloatTensor_sumall(linear1->weight));
        // printf("weight1[0]: %f\n", THFloatTensor_data(linear1->weight)[0]);
        // printf("weight2: %f\n", THFloatTensor_sumall(linear2->weight));
        // printf("weight2[0]: %f\n", THFloatTensor_data(linear2->weight)[0]);

        delete outp_linear1;
        delete outp_sigm;
        delete outp_linear2;
        delete outp_softmax;
        delete outp_nll;

        delete grad_nll;
        delete grad_softmax;
        delete grad_linear2;
        delete grad_sigm;
        delete grad_linear1;

        // Eval
        if (DO_EVAL) {
            inp_linear1 = eval_var;
            outp_linear1 = linear1->forward(inp_linear1);
            inp_sigm = outp_linear1;
            outp_sigm = Sigmoid_forward(inp_sigm);
            inp_linear2 = outp_sigm;
            outp_linear2 = linear2->forward(inp_linear2);
            inp_softmax = outp_linear2;
            outp_softmax = LogSoftMax_forward(inp_softmax);
            inp_nll = outp_softmax;
            outp_nll = NLLLoss_forward(inp_nll, eval_target);
            printf("[%d] eval (nll): %f\n", step, THFloatTensor_sumall(outp_nll->data));

            delete outp_linear1;
            delete outp_sigm;
            delete outp_linear2;
            delete outp_softmax;
            delete outp_nll;
        }
    }

    return 0;
}
