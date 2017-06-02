#include "TH/TH.h"
#include "THNN/THNN.h"

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
        Linear(int inpDim, int outpDim);
        Variable * forward(Variable *x);
        Variable * backward();
        ~Linear();
};

Linear::Linear(int inpDim, int outpDim) {
    this->inpDim = inpDim;
    this->outpDim = outpDim;
    this->weight = THFloatTensor_newWithSize2d(this->outpDim, this->inpDim);
    this->bias = THFloatTensor_newWithSize1d(this->outpDim);
}

Variable * Linear::forward(Variable *x) {
    THFloatTensor *input = x->data;
    THFloatTensor *weight = this->weight;
    THFloatTensor *bias = NULL;
    THNNState *state = NULL;
    THFloatTensor *addBuffer = NULL;
    THFloatTensor *output = THFloatTensor_newWithSize1d(this->outpDim);
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

Variable * Linear::backward() {
    // TODO
    return NULL;
}

int main()
{
    THFile *x_file = THDiskFile_new("x", "r", 0);
    THFile *y_file = THDiskFile_new("y", "r", 0);

    THFloatTensor *x = THFloatTensor_newWithSize1d(10);
    THFloatTensor *y = THFloatTensor_newWithSize1d(10);

    THFile_readFloat(x_file, x->storage);
    THFile_readFloat(y_file, y->storage);

    double result = THFloatTensor_dot(x, y) + THFloatTensor_sumall(x);

    printf("%f\n", result);

    THFloatTensor *z = THFloatTensor_newWithSize1d(10);
    THFloatTensor *zz = THFloatTensor_newWithSize1d(10);
    THNN_FloatLogSigmoid_updateOutput(NULL, x, z, zz);

    printf("%f\n", THFloatTensor_sumall(z));
    printf("%f\n", THFloatTensor_sumall(zz));

    THFloatTensor_free(x);
    THFloatTensor_free(y);
    THFile_free(x_file);
    THFile_free(y_file);

    THFile *batch_file = THDiskFile_new("y", "r", 0);
    THFile *weight_file = THDiskFile_new("w", "r", 0);

    int inp_dim = 10;
    int outp_dim = 10;
    THFloatTensor *batch = THFloatTensor_newWithSize1d(inp_dim);
    Linear *linear = new Linear(inp_dim, outp_dim);

    THFile_readFloat(batch_file, batch->storage);
    THFile_readFloat(weight_file, linear->weight->storage);

    Variable *batch_var = new Variable(batch);
    Variable *outp = linear->forward(batch_var);
    printf("%f\n", THFloatTensor_sumall(outp->data));

    return 0;
}
