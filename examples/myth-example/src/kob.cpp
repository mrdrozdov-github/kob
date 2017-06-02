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

    Variable *result = new Variable(gradInput);
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


void THNN_CustomFloatClassNLLCriterion_updateOutput(
          THNNState *state,
          THFloatTensor *input,
          THIndexTensor *target,
          THFloatTensor *output,
          bool sizeAverage,
          THFloatTensor *weights,
          THFloatTensor *total_weight,
          long ignore_index)
{
  // THNN_CHECK_DIM_SIZE(output, 1, 0, 1);
  // THNN_CHECK_DIM_SIZE(total_weight, 1, 0, 1);
  int n_dims = THFloatTensor_nDimension(input);
  int n_classes = THFloatTensor_size(input, n_dims - 1);
  ignore_index -= TH_INDEX_BASE;

  if (THIndexTensor_(nDimension)(target) > 1) {
    THError("multi-target not supported");
  }
  if (THFloatTensor_nDimension(input) > 2) {
    THError("input tensor should be 1D or 2D");
  }
  if (weights && THFloatTensor_nElement(weights) != n_classes) {
    THDescBuff s1 = THFloatTensor_sizeDesc(weights);
    THError("weight tensor should be defined either for all %d classes or no classes"
        " but got weight tensor of shape: %s", n_classes, s1.str);
  }

  input = THFloatTensor_newContiguous(input);
  target = THIndexTensor_(newContiguous)(target);
  weights = weights ? THFloatTensor_newContiguous(weights) : NULL;

  float *input_data = THFloatTensor_data(input);
  THIndex_t *target_data = THIndexTensor_(data)(target);
  float *weights_data = weights ? THFloatTensor_data(weights) : NULL;
  float *output_data = THFloatTensor_data(output);
  float *total_weight_data = THFloatTensor_data(total_weight);

  output_data[0] = total_weight_data[0] = 0.0;

  if (THFloatTensor_nDimension(input) == 1) {
    int cur_target = target_data[0] - TH_INDEX_BASE;
    if (cur_target != ignore_index) {
      THAssert(cur_target >= 0 && cur_target < n_classes);
      total_weight_data[0] = weights ? weights_data[cur_target] : 1.0f;
      output_data[0] = -input_data[cur_target] * total_weight_data[0];
    }
  } else if (THFloatTensor_nDimension(input) == 2) {
    int batch_size = THFloatTensor_size(input, 0);
    THAssert(THIndexTensor_(size)(target, 0) == batch_size);

    int n_target = THFloatTensor_size(input, 1);

    int i;
    for (i = 0; i < batch_size; i++) {
      int cur_target = target_data[i] - TH_INDEX_BASE;
      if (cur_target != ignore_index) {
        printf("cur_target: %d, n_classes: %d, TH_INDEX_BASE: %d\n", cur_target, n_classes, TH_INDEX_BASE);
        THAssert(cur_target >= 0 && cur_target < n_classes);

        float cur_weight = weights ? weights_data[cur_target] : 1.0f;
        total_weight_data[0] += cur_weight;
        output_data[0] -= input_data[i * n_target + cur_target] * cur_weight;
      }
    }
  }

  if (sizeAverage && total_weight_data[0]) {
    output_data[0] /= total_weight_data[0];
  }

  if (weights) {
    THFloatTensor_free(weights);
  }
  THFloatTensor_free(input);
  THIndexTensor_(free)(target);
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
