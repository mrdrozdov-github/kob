#include "kob.h"

Variable::Variable(THFloatTensor *x) {
    this->data = x;
}

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