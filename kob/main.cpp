#include <iostream>
#include <stdio.h>
#include <math.h>
#include "main.h"

using namespace std;

int main()
{
    Variable *a = new Variable(-1.0);
    Variable *b = new Variable(3.0);
    Variable *c = new Variable(5.0);

    Module *add = new Add();
    Module *mult = new Multiply();
    Module *sigm = new Sigmoid();

    printf("Before:\n");
    printf("a.grad: %f\n", a->grad);
    printf("b.grad: %f\n", b->grad);
    printf("c.grad: %f\n", c->grad);

    Variable *outp1 = add->forward(a, b);
    Variable *outp2 = sigm->forward(outp1);
    Variable *outp3 = mult->forward(outp2, c);

    printf("outp1.value: %f\n", outp1->value);
    printf("outp2.value: %f\n", outp2->value);
    printf("outp3.value: %f\n", outp3->value);

    double loss = 10.0;

    mult->backward(loss);
    sigm->backward(outp2->grad);
    sigm->backward(c->grad);
    add->backward(outp1->grad);
    add->backward(outp1->grad);

    printf("After:\n");
    printf("a.grad: %f\n", a->grad);
    printf("b.grad: %f\n", b->grad);
    printf("c.grad: %f\n", c->grad);

    return 0;
}

/* Variable */

Variable::Variable(double value) {
    this->value = value;
    this->grad = 0.0;
}

/* Add */

Variable * Add::forward(Variable *a, Variable *b) {
    this->a = a;
    this->b = b;
    Variable *result = new Variable(a->value + b->value);
    return result;
}

void Add::backward(double grad) {
    this->a->grad += 1 * grad;
    this->b->grad += 1 * grad;
}

/* Multiply */

Variable * Multiply::forward(Variable *a, Variable *b) {
    this->a = a;
    this->b = b;
    Variable *result = new Variable(a->value * b->value);
    return result;
}

void Multiply::backward(double grad) {
    this->a->grad += this->b->value * grad;
    this->b->grad += this->a->value * grad;
}

/* Sigmoid */

double fn_sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double fn_derivative_sigmoid_helper(double s) {
    return (s * (1 - s));
}

double fn_derivative_sigmoid(double x) {
    double s = fn_sigmoid(x);
    return fn_derivative_sigmoid_helper(s);
}

Variable * Sigmoid::forward(Variable *a) {
    this->a = a;
    this->s = fn_sigmoid(a->value);
    Variable *result = new Variable(this->s);
    return result;
}

void Sigmoid::backward(double grad) {
    this->a->grad += fn_derivative_sigmoid_helper(this->s) * grad;
}
