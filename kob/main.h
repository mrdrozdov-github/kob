#include <iostream>
#include <stdio.h>

using namespace std;

class Variable
{
    public:
        Variable(double value);
        double value;
        double grad;
};

class Module
{
    public:
        virtual Variable * forward(Variable *a) { return NULL; };
        virtual Variable * forward(Variable *a, Variable *b) { return NULL; };
        virtual void backward(double grad) {};
};

class Add : public Module {
    private:
        Variable *a;
        Variable *b;

    public:
        Variable * forward(Variable *a, Variable *b);
        void backward(double grad);
};

class Multiply : public Module {
    private:
        Variable *a;
        Variable *b;

    public:
        Variable * forward(Variable *a, Variable *b);
        void backward(double grad);
};

class Sigmoid : public Module {
    private:
        Variable *a;
        double s;

    public:
        Variable * forward(Variable *a);
        void backward(double grad);
};
