#include <utility>
#include "TH/TH.h"
#include "THNN/THNN.h"

using namespace std;

class Variable;
class Linear;

class Variable
{
    private:
        Variable * recursive_backward(Variable *gradInput);
    public:
        Variable *parent_input;
        THLongTensor *parent_target;
        Linear * parent_linear;
        Variable * (*parent_backward)(Variable *x, THFloatTensor *output, THFloatTensor *gradOutput);
        Variable * (*parent_backward_loss)(Variable *x, THLongTensor *target);
        THFloatTensor *data;
        Variable(THFloatTensor *x);
        ~Variable();
        Variable * backward(Variable *gradOutput);
        Variable * backward();
};

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
        Variable * call(Variable *x);
        Variable * forward(Variable *x);
        Variable * backward(Variable *x, THFloatTensor *gradOutput);
        void clear_grads();
        ~Linear();
};

Variable * F_sigmoid(Variable *x);
Variable * Sigmoid_forward(Variable *x);
Variable * Sigmoid_backward(Variable *x, THFloatTensor *output, THFloatTensor *gradOutput);
Variable * F_log_softmax(Variable *x);
Variable * LogSoftMax_forward(Variable *x);
Variable * LogSoftMax_backward(Variable *x, THFloatTensor *output, THFloatTensor *gradOutput);
Variable * SoftMax_forward(Variable *x);
Variable * SoftMax_backward(Variable *x, THFloatTensor *output, THFloatTensor *gradOutput);
Variable * F_nll(Variable *x, THLongTensor *target);
Variable * NLLLoss_forward(Variable *x, THLongTensor *target);
Variable * NLLLoss_backward(Variable *x, THLongTensor *target);

// Non-differentiable.
pair<Variable *, THLongTensor *> t_Max(Variable *x, int dimension);
THLongTensor * t_Equal(THLongTensor *x, THLongTensor *y);

void readFloat(THFile *file, THFloatTensor *tensor);
void readLong(THFile *file, THLongTensor *tensor);
