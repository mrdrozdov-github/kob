#include "kob.h"

Variable::Variable(THFloatTensor *x) {
    this->parent_backward = nullptr;
    this->parent_linear = nullptr;
    this->parent_input = nullptr;
    this->data = x;
}

Variable::~Variable() {
    THFloatTensor_free(this->data);
}

Variable * Variable::recursive_backward(Variable *gradInput) {
    // Recursively call backward on all Variables.
    if (this->parent_input->parent_backward != nullptr) {
        gradInput = this->parent_input->backward(gradInput);
        THFloatTensor_free(gradInput->data);
    } else if (this->parent_input->parent_linear != nullptr) {
        gradInput = this->parent_input->backward(gradInput);
        THFloatTensor_free(gradInput->data);
    }

    return gradInput;
}

Variable * Variable::backward(Variable *gradOutput) {
    Variable *gradInput;
    if (this->parent_backward != nullptr) {
        gradInput = this->parent_backward(this->parent_input, this->data, gradOutput->data);
        THFloatTensor_free(this->data);
    }
    else if (this->parent_linear != nullptr) {
        gradInput = this->parent_linear->backward(this->parent_input, gradOutput->data);
        THFloatTensor_free(this->data);
    }
    return this->recursive_backward(gradInput);
}

Variable * Variable::backward() {
    Variable *gradInput = this->parent_backward_loss(this->parent_input, this->parent_target);
    THFloatTensor_free(this->data);
    return this->recursive_backward(gradInput);
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

Variable * Linear::call(Variable *x) {
    Variable *result = this->forward(x);
    result->parent_input = x;
    result->parent_linear = this;
    return result;
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

    THFloatTensor_free(addBuffer);

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

    // Calculate gradient with respect to input.
    THNN_FloatLinear_updateGradInput(
          state,
          input,
          gradOutput,
          gradInput,
          weight);

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

    THFloatTensor_free(addBuffer);

    Variable *result = new Variable(gradInput);
    return result;
}

Variable * F_sigmoid(Variable *x) {
    Variable *result = Sigmoid_forward(x);
    result->parent_input = x;
    result->parent_backward = Sigmoid_backward;
    return result;
}

Variable * Sigmoid_forward(Variable *x) {
    long batch_size = x->data->size[0];
    long dim_size = x->data->size[1];

    THFloatTensor *input = x->data;
    THFloatTensor *output = THFloatTensor_newWithSize2d(batch_size, dim_size);
    THFloatTensor_zero(output);
    THNNState *state = NULL;

    THNN_FloatSigmoid_updateOutput(
          state,
          input,
          output);

    Variable *result = new Variable(output);
    return result;
}

Variable * Sigmoid_backward(Variable *x, THFloatTensor *output, THFloatTensor *gradOutput) {
    long batch_size = x->data->size[0];
    long dim_size = x->data->size[1];

    THFloatTensor *input = x->data;
    THFloatTensor *gradInput = THFloatTensor_newWithSize2d(batch_size, dim_size);
    THNNState *state = NULL;
    THFloatTensor_zero(gradInput);

    THNN_FloatSigmoid_updateGradInput(
          state,
          input,
          gradOutput,
          gradInput,
          output);

    Variable *result = new Variable(gradInput);
    return result;
}

Variable * F_log_softmax(Variable *x) {
    Variable *result = LogSoftMax_forward(x);
    result->parent_input = x;
    result->parent_backward = LogSoftMax_backward;
    return result;
}

Variable * LogSoftMax_forward(Variable *x) {
    long batch_size = x->data->size[0];
    long dim_size = x->data->size[1];

    THFloatTensor *input = x->data;
    THFloatTensor *output = THFloatTensor_newWithSize2d(batch_size, dim_size);
    THFloatTensor_zero(output);
    THNNState *state = NULL;

    THNN_FloatLogSoftMax_updateOutput(
          state,
          input,
          output);

    Variable *result = new Variable(output);
    return result;
}

Variable * LogSoftMax_backward(Variable *x, THFloatTensor *output, THFloatTensor *gradOutput) {
    long batch_size = x->data->size[0];
    long dim_size = x->data->size[1];

    THFloatTensor *input = x->data;
    THFloatTensor *gradInput = THFloatTensor_newWithSize2d(batch_size, dim_size);
    THNNState *state = NULL;
    THFloatTensor_zero(gradInput);

    THNN_FloatLogSoftMax_updateGradInput(
          state,
          input,
          gradOutput,
          gradInput,
          output);

    Variable *result = new Variable(gradInput);
    return result;
}

Variable * SoftMax_forward(Variable *x) {
    long batch_size = x->data->size[0];
    long dim_size = x->data->size[1];

    THFloatTensor *input = x->data;
    THFloatTensor *output = THFloatTensor_newWithSize2d(batch_size, dim_size);
    THFloatTensor_zero(output);
    THNNState *state = NULL;

    THNN_FloatSoftMax_updateOutput(
          state,
          input,
          output);

    Variable *result = new Variable(output);
    return result;
}

Variable * SoftMax_backward(Variable *x, THFloatTensor *output, THFloatTensor *gradOutput) {
    long batch_size = x->data->size[0];
    long dim_size = x->data->size[1];

    THFloatTensor *input = x->data;
    THFloatTensor *gradInput = THFloatTensor_newWithSize2d(batch_size, dim_size);
    THNNState *state = NULL;
    THFloatTensor_zero(gradInput);

    THNN_FloatSoftMax_updateGradInput(
          state,
          input,
          gradOutput,
          gradInput,
          output);

    Variable *result = new Variable(gradInput);
    return result;
}

Variable * F_nll(Variable *x, THLongTensor *target) {
    Variable *result = NLLLoss_forward(x, target);
    result->parent_input = x;
    result->parent_target = target;
    result->parent_backward_loss = NLLLoss_backward;
    return result;
}

Variable * NLLLoss_forward(Variable *x, THLongTensor *target) {
    long batch_size = x->data->size[0];
    long dim_size = x->data->size[1];

    THFloatTensor *input = x->data;
    THFloatTensor *output = THFloatTensor_newWithSize1d(1);
    THFloatTensor_zero(output);
    THNNState *state = NULL;
    bool sizeAverage = true;
    THFloatTensor *weights = NULL;
    THFloatTensor *total_weight = THFloatTensor_newWithSize1d(1);
    THFloatTensor_fill(total_weight, 1.0);
    long ignore_index = -1;

    THNN_FloatClassNLLCriterion_updateOutput(
          state,
          input,
          target,
          output,
          sizeAverage,
          weights,
          total_weight,
          ignore_index);

    THFloatTensor_free(total_weight);

    Variable *result = new Variable(output);
    return result;
}

Variable * NLLLoss_backward(Variable *x, THLongTensor *target) {
    long batch_size = x->data->size[0];
    long dim_size = x->data->size[1];

    THFloatTensor *input = x->data;
    THFloatTensor *gradInput = THFloatTensor_newWithSize2d(batch_size, dim_size);
    THNNState *state = NULL;
    THFloatTensor_zero(gradInput);
    bool sizeAverage = true;
    THFloatTensor *weights = NULL;
    THFloatTensor *total_weight = THFloatTensor_newWithSize1d(1);
    THFloatTensor_fill(total_weight, 1.0);
    long ignore_index = -1;

    THNN_FloatClassNLLCriterion_updateGradInput(
          state,
          input,
          target,
          gradInput,
          sizeAverage,
          weights,
          total_weight,
          ignore_index);

    THFloatTensor_free(total_weight);

    Variable *result = new Variable(gradInput);
    return result;
}

// Non-differentiable.

pair<Variable *, THLongTensor *> t_Max(Variable *x, int dimension) {
    THFloatTensor *data = x->data;
    THFloatTensor *_values = THFloatTensor_newWithSize2d(data->size[0], 1);
    THLongTensor *indices = THLongTensor_newWithSize2d(data->size[0], 1);
    int keepdim = 1;

    THFloatTensor_max(_values, indices, data, dimension, keepdim);

    Variable *values = new Variable(_values);
    pair<Variable *, THLongTensor *> result = make_pair(values, indices);

    return result;
}

THLongTensor * t_Equal(THLongTensor *x, THLongTensor *y) {
    THLongTensor *result = THLongTensor_newWithSize2d(x->size[0], y->size[1]);
    THLongTensor_eqTensorT(result, x, y);
    return result;
}

// Utility.

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

void readLong(THFile *file, THLongTensor *tensor) {
    THLongStorage *size = THLongTensor_newSizeOf(tensor);
    THLongStorage *stride = THLongTensor_newStrideOf(tensor);
    ptrdiff_t numel = THLongTensor_nElement(tensor);

    // Flatten
    THLongTensor_resize1d(tensor, numel);
    // Read Data
    THFile_readLong(file, tensor->storage);
    // Restore Original Size
    THLongTensor_resize(tensor, size, stride);

    THLongStorage_free(size);
    THLongStorage_free(stride);
}
