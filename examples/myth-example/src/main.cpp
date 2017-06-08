#include <math.h>
#include <iostream>

#include "kob.h"
#include "batch_reader.h"
#include "gflags/gflags.h"

#define PRINT_SAMPLE false

using namespace std;

/*

TODO:

- [x] batching
- [ ] don't need to specify data size
- [ ] pretty logging (w. timestamps)
- [ ] protobuf for statistics
- [x] evaluation
- [ ] benchmark against pytorch (accuracy/loss)
- [ ] benchmark against pytorch (speed)
- [x] log accuracy in eval
- [ ] weight initialization
- [ ] auto-backward

*/

DEFINE_string(train_file, "/Users/Andrew/Developer/kob/examples/h5mnist/train.h5", "Data file");
DEFINE_string(eval_file, "/Users/Andrew/Developer/kob/examples/h5mnist/test.h5", "Data file");
DEFINE_string(weight1_file, "w1", "Data file");
DEFINE_string(weight2_file, "w2", "Data file");
DEFINE_int32(data_size, 60000, "Data dim");
DEFINE_int32(eval_data_size, 100, "Data dim");
DEFINE_int32(batch_size, 100, "Data dim");
DEFINE_int32(inp_dim, 784, "Data dim");
DEFINE_int32(hidden_dim, 64, "Data dim");
DEFINE_int32(outp_dim, 10, "Data dim");
DEFINE_int32(steps, 1, "Data dim");
DEFINE_int32(seed, 11, "Random seed");
DEFINE_int32(epochs, 10, "Number of epochs");
DEFINE_double(learning_rate, 0.001, "Data dim");
DEFINE_bool(run_eval, false, "Run eval once per epoch");
DEFINE_bool(verbose, false, "Verbose");

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

void run_eval(Linear *linear1, Linear *linear2) {
    string filename = FLAGS_eval_file;
    int n = 10000;
    int correct = 0;
    int size = FLAGS_inp_dim;
    int batch_size = FLAGS_batch_size;
    int num_batches = n / batch_size;

    BatchReader batch_reader = BatchReader(filename, "images", n, size);
    BatchReader batch_reader_labels = BatchReader(filename, "labels", n, 1);

    H5File file(filename, H5F_ACC_RDONLY);
    DataSet dataset_images = file.openDataSet("images");
    DataSet dataset_labels = file.openDataSet("labels");

    THFloatTensor *batch = THFloatTensor_newWithSize2d(FLAGS_batch_size, size);
    THLongTensor *target = THLongTensor_newWithSize1d(FLAGS_batch_size);

    Variable *inp_linear1;
    Variable *outp_linear1;
    Variable *inp_sigm;
    Variable *outp_sigm;
    Variable *inp_linear2;
    Variable *outp_linear2;
    Variable *inp_softmax;
    Variable *outp_softmax;
    pair<Variable *, THLongTensor *> outp_max;
    THLongTensor *outp_eq;

    Variable *var = new Variable(batch);

    int index[n];
    for (int i=0; i<n; ++i) {
        index[i] = i;
    }

    for (int i_batch = 0; i_batch < num_batches; ++i_batch) {
        // Prepare batch.
        batch_reader.read_batch(THFloatTensor_data(var->data), index + i_batch * batch_size, batch_size, file, dataset_images);
        batch_reader_labels.read_batch(THLongTensor_data(target), index + i_batch * batch_size, batch_size, file, dataset_labels);

        // Fix labels.
        THLongTensor_add(target, target, 1);

        inp_linear1 = var;
        outp_linear1 = linear1->forward(inp_linear1);
        inp_sigm = outp_linear1;
        outp_sigm = Sigmoid_forward(inp_sigm);
        inp_linear2 = outp_sigm;
        outp_linear2 = linear2->forward(inp_linear2);
        inp_softmax = outp_linear2;
        outp_softmax = SoftMax_forward(inp_softmax);
        outp_max = t_Max(outp_softmax, 1);

        THLongTensor_sub(target, target, 1);
        outp_eq = t_Equal(target, outp_max.second);

        correct += THLongTensor_sumall(outp_eq);

    }
    printf("Eval: Correct: %d\n", correct);

    delete outp_linear1;
    delete outp_sigm;
    delete outp_linear2;
    delete outp_softmax;
    delete outp_max.first;
    delete outp_max.second;
    delete outp_eq;
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

    string filename = FLAGS_train_file;
    int n = FLAGS_data_size;
    int size = FLAGS_inp_dim;
    int batch_size = FLAGS_batch_size;
    int num_batches = n / batch_size;

    BatchReader batch_reader = BatchReader(filename, "images", n, size);
    BatchReader batch_reader_labels = BatchReader(filename, "labels", n, 1);

    H5File file(filename, H5F_ACC_RDONLY);
    DataSet dataset_images = file.openDataSet("images");
    DataSet dataset_labels = file.openDataSet("labels");

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
    Variable *grad_input;

    for (int i_epoch = 0; i_epoch < FLAGS_epochs; ++i_epoch) {
        // Run one epoch.
        random_shuffle(index, index+n);

        for (int i_batch = 0; i_batch < num_batches; ++i_batch) {
            THFloatTensor *batch = THFloatTensor_newWithSize2d(FLAGS_batch_size, size);
            THLongTensor *target = THLongTensor_newWithSize1d(FLAGS_batch_size);
            Variable *train_var = new Variable(batch);

            // Prepare batch.
            batch_reader.read_batch(THFloatTensor_data(train_var->data), index + i_batch * batch_size, batch_size, file, dataset_images);
            batch_reader_labels.read_batch(THLongTensor_data(target), index + i_batch * batch_size, batch_size, file, dataset_labels);

            // Fix labels.
            THLongTensor_add(target, target, 1);

            // Forward pass.
            inp_linear1 = train_var;
            outp_linear1 = linear1->call(inp_linear1);

            inp_sigm = outp_linear1;
            outp_sigm = F_sigmoid(inp_sigm);

            inp_linear2 = outp_sigm;
            outp_linear2 = linear2->call(inp_linear2);

            inp_softmax = outp_linear2;
            outp_softmax = F_log_softmax(inp_softmax);

            inp_nll = outp_softmax;
            outp_nll = F_nll(inp_nll, target);
            if (FLAGS_verbose) {
                printf("Train Epoch: %d [%d/%d (%.0f%%)]\tLoss: %.6f\n",
                    i_epoch, i_batch * batch_size, n,
                    100. * i_batch / num_batches, THFloatTensor_sumall(outp_nll->data));
            }

            // Backward Pass
            linear1->clear_grads();
            linear2->clear_grads();

            // Backward Pass
            grad_input = outp_nll->backward();

            // Gradient update.
            THFloatTensor_csub(linear1->weight, linear1->weight, FLAGS_learning_rate, linear1->gradWeight);
            THFloatTensor_csub(linear2->weight, linear2->weight, FLAGS_learning_rate, linear2->gradWeight);

            delete outp_linear1;
            delete outp_sigm;
            delete outp_linear2;
            delete outp_softmax;
            delete outp_nll;

            delete grad_input;
        }

        if (FLAGS_run_eval) {
            run_eval(linear1, linear2);
        }
    }

    return 0;
}
