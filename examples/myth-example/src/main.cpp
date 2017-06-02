#include "TH/TH.h"
#include "THNN/THNN.h"
#include "gflags/gflags.h"

class Variable
{
    private:
    public:
        THFloatTensor *data;
        Variable(THFloatTensor *x);
        ~Variable();
    
};

Variable::Variable(THFloatTensor *x) {
    this->data = x;
}

/* Custom Linear Layer 

Please compare:

https://github.com/torch/nn/blob/master/lib/THNN/generic/Linear.c

to

https://github.com/tuotuoxp/cpp-torch/blob/master/src/nn/Linear.h.inl

*/

class Linear
{
    private:
    public:
        int inpDim;
        int outpDim;
        THFloatTensor *weight;
        THFloatTensor *bias;
        THFloatTensor *gradWeight;
        THFloatTensor *gradBias;
        Linear(int inpDim, int outpDim);
        Variable * forward(Variable *x);
        Variable * backward(Variable *x, THFloatTensor *gradOutput);
        void clear_grads();
        ~Linear();
};

Linear::Linear(int inpDim, int outpDim) {
    this->inpDim = inpDim;
    this->outpDim = outpDim;
    this->weight = THFloatTensor_newWithSize2d(this->outpDim, this->inpDim);
    this->gradWeight = THFloatTensor_newWithSize2d(this->outpDim, this->inpDim);
    this->bias = NULL;
    this->gradBias = NULL;

    // this->bias = THFloatTensor_newWithSize1d(this->outpDim);
    // this->gradBias = THFloatTensor_newWithSize1d(this->outpDim);
    // THFloatTensor_zero(this->bias);
}

void Linear::clear_grads() {
    THFloatTensor_zero(this->gradWeight);
    // THFloatTensor_zero(this->gradBias);
}

Variable * Linear::forward(Variable *x) {
    int dim = x->data->nDimension;
    THFloatTensor *input = x->data;
    THFloatTensor *weight = this->weight;
    THFloatTensor *bias = this->bias;
    THNNState *state = NULL;
    THFloatTensor *addBuffer;
    THFloatTensor *output;

    if (dim == 1) {
        output = THFloatTensor_newWithSize1d(this->outpDim);
    } else if (dim == 2) {
        long batch_size = x->data->size[0];
        addBuffer = THFloatTensor_newWithSize1d(batch_size);
        output = THFloatTensor_newWithSize2d(batch_size, this->outpDim);
    }

    THNN_FloatLinear_updateOutput(
        state,
        input,
        output,
        weight,
        bias,
        addBuffer);
    Variable *result = new Variable(output);
    return result;
}

Variable * Linear::backward(Variable *x, THFloatTensor *gradOutput) {
    long batch_size = x->data->size[0];
    THFloatTensor *input = x->data;
    THFloatTensor *weight = this->weight;
    THFloatTensor *bias = this->bias;
    THNNState *state = NULL;
    THFloatTensor *addBuffer = THFloatTensor_newWithSize1d(batch_size);
    float scale = 1;

    THFloatTensor *gradInput = THFloatTensor_newWithSize2d(batch_size, this->inpDim); // TODO
    THFloatTensor_zero(gradInput);

    THFloatTensor *gradWeight = this->gradWeight; // TODO
    THFloatTensor *gradBias = this->gradBias; // TODO

    printf("updateGradInput\n");
    // Calculate gradient with respect to input.
    THNN_FloatLinear_updateGradInput(
          state,
          input,
          gradOutput,
          gradInput,
          weight);

    printf("accGradParameters\n");
    // Done for layers that have parameters.
    THNN_FloatLinear_accGradParameters(
          state,
          input,
          gradOutput,
          gradInput,
          weight,
          bias,
          gradWeight,
          gradBias,
          addBuffer,
          scale);

    Variable *result = new Variable(gradInput);
    return result;
}

void readFloat(THFile *file, THFloatTensor *tensor) {
    THLongStorage *size = THFloatTensor_newSizeOf(tensor);
    THLongStorage *stride = THFloatTensor_newStrideOf(tensor);
    ptrdiff_t numel = THFloatTensor_nElement(tensor);

    // Flatten
    THFloatTensor_resize1d(tensor, numel);
    // Read Data
    THFile_readFloat(file, tensor->storage);
    // Restore Original Size
    THFloatTensor_resize(tensor, size, stride);

    THLongStorage_free(size);
    THLongStorage_free(stride);
}

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
    Variable *outp = linear->forward(batch_var);
    printf("forward: %f\n", THFloatTensor_sumall(outp->data));

    // Backward Pass
    THFloatTensor *loss = THFloatTensor_newWithSize2d(batch_size, outp_dim);
    THFloatTensor_fill(loss, 1.0);
    printf("loss: %f\n", THFloatTensor_sumall(loss));
    linear->clear_grads();
    printf("grads: %f\n", THFloatTensor_sumall(linear->gradWeight));
    Variable *gradInput = linear->backward(outp, loss);
    printf("gradInput: %f\n", THFloatTensor_sumall(gradInput->data));
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
