#include "kob.h"
#include "gflags/gflags.h"

DEFINE_string(batch_file, "y", "Data file");
DEFINE_string(weight1_file, "w1", "Data file");
DEFINE_string(weight2_file, "w2", "Data file");
DEFINE_int32(batch_size, 2, "Data dim");
DEFINE_int32(inp_dim, 10, "Data dim");
DEFINE_int32(outp_dim, 10, "Data dim");
DEFINE_double(learning_rate, 0.001, "Data dim");

int main(int argc, char *argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    gflags::ShutDownCommandLineFlags();

    THFile *batch_file = THDiskFile_new(FLAGS_batch_file.c_str(), "r", 0);
    THFile *weight1_file = THDiskFile_new(FLAGS_weight1_file.c_str(), "r", 0);
    THFile *weight2_file = THDiskFile_new(FLAGS_weight2_file.c_str(), "r", 0);

    int batch_size = FLAGS_batch_size;
    int inp_dim = FLAGS_inp_dim;
    int outp_dim = FLAGS_outp_dim;
    THFloatTensor *batch = THFloatTensor_newWithSize2d(batch_size, inp_dim);
    Linear *linear1 = new Linear(inp_dim, outp_dim);
    Linear *linear2 = new Linear(outp_dim, outp_dim);

    // Initialization
    readFloat(batch_file, batch);
    readFloat(weight1_file, linear1->weight);
    readFloat(weight2_file, linear2->weight);
    printf("weight1: %f\n", THFloatTensor_sumall(linear1->weight));
    printf("weight1[0]: %f\n", THFloatTensor_data(linear1->weight)[0]);
    printf("weight2: %f\n", THFloatTensor_sumall(linear2->weight));
    printf("weight2[0]: %f\n", THFloatTensor_data(linear2->weight)[0]);

    // Forward Pass
    Variable *batch_var = new Variable(batch);

    Variable *inp_linear1 = batch_var;
    Variable *outp_linear1 = linear1->forward(batch_var);
    printf("forward (linear1): %f\n", THFloatTensor_sumall(outp_linear1->data));

    Variable *inp_sigm = outp_linear1;
    Variable *outp_sigm = Sigmoid_forward(inp_sigm);
    printf("forward (sigm): %f\n", THFloatTensor_sumall(outp_sigm->data));

    Variable *inp_linear2 = outp_sigm;
    Variable *outp_linear2 = Sigmoid_forward(inp_linear2);
    printf("forward (linear2): %f\n", THFloatTensor_sumall(outp_linear2->data));

    // Backward Pass
    THFloatTensor *loss = THFloatTensor_newWithSize2d(batch_size, outp_dim);
    THFloatTensor_fill(loss, 1.0);
    THFloatTensor_csub(loss, loss, 1.0, outp_linear2->data);
    printf("loss: %f\n", THFloatTensor_sumall(loss));
    linear1->clear_grads();
    linear2->clear_grads();

    printf("grads (linear1): %f\n", THFloatTensor_sumall(linear1->gradWeight));
    printf("grads (linear2): %f\n", THFloatTensor_sumall(linear2->gradWeight));
    printf("grads (linear1)[0]: %f\n", THFloatTensor_data(linear1->gradWeight)[0]);
    printf("grads (linear2)[0]: %f\n", THFloatTensor_data(linear2->gradWeight)[0]);

    Variable *grad_linear2 = linear2->backward(inp_linear2, loss);
    printf("grad (linear2): %f\n", THFloatTensor_sumall(grad_linear2->data));
    printf("grad (linear2)(numel): %td\n", THFloatTensor_nElement(grad_linear2->data));

    Variable *grad_sigm = Sigmoid_backward(inp_sigm, outp_sigm->data, grad_linear2->data);
    printf("grad (sigm): %f\n", THFloatTensor_sumall(grad_sigm->data));
    printf("grad (sigm)(numel): %td\n", THFloatTensor_nElement(grad_sigm->data));

    Variable *grad_linear1 = linear1->backward(inp_linear1, grad_sigm->data);
    printf("grad (linear1): %f\n", THFloatTensor_sumall(grad_linear1->data));
    printf("grad (linear1)(numel): %td\n", THFloatTensor_nElement(grad_linear1->data));

    printf("grads (linear1): %f\n", THFloatTensor_sumall(linear1->gradWeight));
    printf("grads (linear1)[0]: %f\n", THFloatTensor_data(linear1->gradWeight)[0]);
    printf("grads (linear2): %f\n", THFloatTensor_sumall(linear2->gradWeight));
    printf("grads (linear2)[0]: %f\n", THFloatTensor_data(linear2->gradWeight)[0]);

    THFloatTensor_csub(linear1->weight, linear1->weight, FLAGS_learning_rate, linear1->gradWeight);
    THFloatTensor_csub(linear2->weight, linear2->weight, FLAGS_learning_rate, linear2->gradWeight);
    printf("weight1: %f\n", THFloatTensor_sumall(linear1->weight));
    printf("weight1[0]: %f\n", THFloatTensor_data(linear1->weight)[0]);
    printf("weight2: %f\n", THFloatTensor_sumall(linear2->weight));
    printf("weight2[0]: %f\n", THFloatTensor_data(linear2->weight)[0]);

    THFloatTensor_free(batch);
    THFile_free(batch_file);
    THFile_free(weight1_file);
    THFile_free(weight2_file);

    return 0;
}
