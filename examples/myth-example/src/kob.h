#include "TH/TH.h"
#include "THNN/THNN.h"

class Variable;
class Linear;

class Variable
{
    private:
    public:
        THFloatTensor *data;
        Variable(THFloatTensor *x);
        ~Variable();
    
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
        Variable * forward(Variable *x);
        Variable * backward(Variable *x, THFloatTensor *gradOutput);
        void clear_grads();
        ~Linear();
};

Variable * Sigmoid_forward(Variable *x);
Variable * Sigmoid_backward(Variable *x, THFloatTensor *output, THFloatTensor *gradOutput);

void readFloat(THFile *file, THFloatTensor *tensor);
