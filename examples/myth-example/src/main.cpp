#include <math.h>
#include <iostream>

#include "kob.h"
#include "batch_reader.h"
#include "gflags/gflags.h"

#define DO_EVAL false
#define PRINT_SAMPLE false

using namespace std;

/*

TODO:

- [x] batching
- [ ] don't need to specify data size
- [ ] pretty logging (w. timestamps)
- [ ] protobuf for statistics
- [ ] evaluation interval
- [ ] benchmark against pytorch (accuracy/loss)
- [ ] benchmark against pytorch (speed)
- [ ] log accuracy
- [ ] weight initialization

*/

DEFINE_string(train_data_file, "train_data.txt", "Data file");
DEFINE_string(train_labels_file, "train_labels.txt", "Data file");
DEFINE_string(eval_data_file, "test_data.txt", "Data file");
DEFINE_string(eval_labels_file, "test_labels.txt", "Data file");
DEFINE_string(weight1_file, "w1", "Data file");
DEFINE_string(weight2_file, "w2", "Data file");
DEFINE_int32(data_size, 100, "Data dim");
DEFINE_int32(eval_data_size, 100, "Data dim");
DEFINE_int32(batch_size, 100, "Data dim");
DEFINE_int32(inp_dim, 784, "Data dim");
DEFINE_int32(hidden_dim, 64, "Data dim");
DEFINE_int32(outp_dim, 10, "Data dim");
DEFINE_int32(steps, 1, "Data dim");
DEFINE_int32(seed, 11, "Random seed");
DEFINE_int32(epochs, 10, "Number of epochs");
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
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    gflags::ShutDownCommandLineFlags();

    int inp_dim = FLAGS_inp_dim;
    int hidden_dim = FLAGS_hidden_dim;
    int outp_dim = FLAGS_outp_dim;

    // Prepare data.
    srand(FLAGS_seed);

    string filename = "/Users/Andrew/Developer/kob/examples/h5mnist/train.h5";
    int n = 55000;
    int size = 784;
    int batch_size = FLAGS_batch_size;
    int num_batches = n / batch_size;

    H5File file(filename, H5F_ACC_RDONLY);
    DataSet dataset_images = file.openDataSet("images");
    DataSet dataset_labels = file.openDataSet("labels");

    BatchReader batch_reader = BatchReader(filename, "images", n, size);
    BatchReader batch_reader_labels = BatchReader(filename, "labels", n, 1);

    THFloatTensor *batch = THFloatTensor_newWithSize2d(FLAGS_batch_size, size);
    THLongTensor *target = THLongTensor_newWithSize1d(FLAGS_batch_size);

    int index[n];
    for (int i=0; i<n; ++i) {
        index[i] = i;
    }

    // Prepare model.
    Linear *linear1 = new Linear(inp_dim, hidden_dim);
    Linear *linear2 = new Linear(hidden_dim, outp_dim);

    THFile *weight1_file = THDiskFile_new(FLAGS_weight1_file.c_str(), "r", 0);
    THFile *weight2_file = THDiskFile_new(FLAGS_weight2_file.c_str(), "r", 0);

    printf("Reading linear1\n");
    readFloat(weight1_file, linear1->weight);
    printf("Reading linear2\n");
    readFloat(weight2_file, linear2->weight);

    THFile_free(weight1_file);
    THFile_free(weight2_file);

    Variable *train_var = new Variable(batch);

    // Forward variables.
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

    // Backward variables.
    Variable *grad_nll;
    Variable *grad_softmax;
    Variable *grad_linear2;
    Variable *grad_sigm;
    Variable *grad_linear1;

    // Print sample of data.
    if (PRINT_SAMPLE) {
        for (int i_batch = 0; i_batch < num_batches; i_batch++) {
            printf("Batch: %d\n", i_batch);
            batch_reader.read_batch(THFloatTensor_data(train_var->data), index + i_batch * batch_size, batch_size, file, dataset_images);
            batch_reader_labels.read_batch(THLongTensor_data(target), index + i_batch * batch_size, batch_size, file, dataset_labels);
            if (i_batch == 0) {
                for (int ii = 0; ii<batch_size; ++ii) {
                    printf("Label: %ld\n", THLongTensor_data(target)[ii]);
                    print_mnist(THFloatTensor_data(train_var->data) + ii * size);
                }
            }
        }
    }

    for (int i_epoch = 0; i_epoch < FLAGS_epochs; ++i_epoch) {
        // Run one epoch.
        random_shuffle(index, index+n);

        for (int i_batch = 0; i_batch < num_batches; ++i_batch) {
            // Prepare batch.
            batch_reader.read_batch(THFloatTensor_data(train_var->data), index + i_batch * batch_size, batch_size, file, dataset_images);
            batch_reader_labels.read_batch(THLongTensor_data(target), index + i_batch * batch_size, batch_size, file, dataset_labels);

            // Fix labels.
            THLongTensor_add(target, target, 1);

            // Forward pass.
            inp_linear1 = train_var;
            outp_linear1 = linear1->forward(inp_linear1);

            inp_sigm = outp_linear1;
            outp_sigm = Sigmoid_forward(inp_sigm);

            inp_linear2 = outp_sigm;
            outp_linear2 = linear2->forward(inp_linear2);

            inp_softmax = outp_linear2;
            outp_softmax = LogSoftMax_forward(inp_softmax);

            inp_nll = outp_softmax;
            outp_nll = NLLLoss_forward(inp_nll, target);
            printf("[epoch=%d, batch=%d] forward (nll): %f\n", i_epoch, i_batch, THFloatTensor_sumall(outp_nll->data));

            // Backward Pass
            linear1->clear_grads();
            linear2->clear_grads();

            // Backward Pass
            grad_nll = NLLLoss_backward(inp_nll, target);
            grad_softmax = LogSoftMax_backward(inp_softmax, outp_softmax->data, grad_nll->data);
            grad_linear2 = linear2->backward(inp_linear2, grad_softmax->data);
            grad_sigm = Sigmoid_backward(inp_sigm, outp_sigm->data, grad_linear2->data);
            grad_linear1 = linear1->backward(inp_linear1, grad_sigm->data);

            // Gradient update.
            THFloatTensor_csub(linear1->weight, linear1->weight, FLAGS_learning_rate, linear1->gradWeight);
            THFloatTensor_csub(linear2->weight, linear2->weight, FLAGS_learning_rate, linear2->gradWeight);

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
        }
    }

    return 0;

    //     // Eval
    //     if (DO_EVAL) {
    //         inp_linear1 = eval_var;
    //         outp_linear1 = linear1->forward(inp_linear1);
    //         inp_sigm = outp_linear1;
    //         outp_sigm = Sigmoid_forward(inp_sigm);
    //         inp_linear2 = outp_sigm;
    //         outp_linear2 = linear2->forward(inp_linear2);
    //         inp_softmax = outp_linear2;
    //         outp_softmax = LogSoftMax_forward(inp_softmax);
    //         inp_nll = outp_softmax;
    //         outp_nll = NLLLoss_forward(inp_nll, eval_target);
    //         printf("[%d] eval (nll): %f\n", step, THFloatTensor_sumall(outp_nll->data));

    //         delete outp_linear1;
    //         delete outp_sigm;
    //         delete outp_linear2;
    //         delete outp_softmax;
    //         delete outp_nll;
    //     }
    // }

    // return 0;
}
