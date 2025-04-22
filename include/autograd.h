/**
 * @file autograd.h
 * @author Eli Asmar
 * @brief Defines the Function base class and operation subclasses used for automatic differentiation.
 */

#ifndef TENSOR_LITE_AUTOGRAD_H
#define TENSOR_LITE_AUTOGRAD_H

#include <memory>
#include <vector>

class Tensor; // forward declaration

struct Function {
    std::vector<Tensor> inputs;

    virtual Tensor forward(const std::vector<Tensor>& inputs_) = 0;
    virtual std::vector<Tensor> backward(const Tensor& grad_output) = 0;

    virtual ~Function() = default;
};

struct Add : public Function {
    Tensor forward(const std::vector<Tensor>& inputs_) override;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

struct Multiply : public Function{
    Tensor forward(const std::vector<Tensor>& inputs_) override;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

struct Subtract : public Function
{
    Tensor forward(const std::vector<Tensor>& inputs_) override;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};


struct Divide : public Function
{
    Tensor forward(const std::vector<Tensor>& inputs_) override;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};


#endif // TENSOR_LITE_AUTOGRAD_H
