#include "kob.h"
#include "gflags/gflags.h"

DEFINE_string(batch_file, "y", "Data file");
DEFINE_string(weight_file, "w", "Data file");
DEFINE_int32(batch_size, 2, "Data dim");
DEFINE_int32(inp_dim, 10, "Data dim");
DEFINE_int32(outp_dim, 10, "Data dim");
DEFINE_double(learning_rate, 0.001, "Data dim");

int main(int argc, char *argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    gflags::ShutDownCommandLineFlags();

    THFile *batch_file = THDiskFile_new(FLAGS_batch_file.c_str(), "r", 0);
    THFile *weight_file = THDiskFile_new(FLAGS_weight_file.c_str(), "r", 0);

    int batch_size = FLAGS_batch_size;
    int inp_dim = FLAGS_inp_dim;
    int outp_dim = FLAGS_outp_dim;
    THFloatTensor *batch = THFloatTensor_newWithSize2d(batch_size, inp_dim);
    Linear *linear = new Linear(inp_dim, outp_dim);

    // Initialization
    readFloat(batch_file, batch);
    readFloat(weight_file, linear->weight);
    printf("weight: %f\n", THFloatTensor_sumall(linear->weight));
    printf("weight[0]: %f\n", THFloatTensor_data(linear->weight)[0]);

    // Forward Pass
    Variable *batch_var = new Variable(batch);
    Variable *outp_linear = linear->forward(batch_var);
    printf("forward: %f\n", THFloatTensor_sumall(outp_linear->data));

    Variable *outp_sigm = Sigmoid_forward(outp_linear);
    printf("forward: %f\n", THFloatTensor_sumall(outp_sigm->data));

    // Backward Pass
    THFloatTensor *loss = THFloatTensor_newWithSize2d(batch_size, outp_dim);
    THFloatTensor_fill(loss, 1.0);
    printf("loss: %f\n", THFloatTensor_sumall(loss));
    linear->clear_grads();
    printf("grads: %f\n", THFloatTensor_sumall(linear->gradWeight));

    Variable *grad_sigm = linear->backward(outp_sigm, loss);
    printf("gradInput: %f\n", THFloatTensor_sumall(grad_sigm->data));

    Variable *grad_linear = linear->backward(outp_linear, grad_sigm->data);
    printf("gradInput: %f\n", THFloatTensor_sumall(grad_linear->data));
    printf("grads: %f\n", THFloatTensor_sumall(linear->gradWeight));
    printf("grads[0]: %f\n", THFloatTensor_data(linear->gradWeight)[0]);

    THFloatTensor_csub(linear->weight, linear->weight, FLAGS_learning_rate, linear->gradWeight);
    printf("weight: %f\n", THFloatTensor_sumall(linear->weight));
    printf("weight[0]: %f\n", THFloatTensor_data(linear->weight)[0]);

    THFloatTensor_free(batch);
    THFile_free(batch_file);
    THFile_free(weight_file);

    return 0;
}
