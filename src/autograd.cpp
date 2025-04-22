/**
 * @file autograd.cpp
 * @author Eli Asmar
 */

#include "../include/autograd.h"
#include "../include/tensor.h"




Tensor Add::forward(const std::vector<Tensor>& inputs_)
{
    inputs = inputs_;
    // Create result tensor with same shape
    Tensor result(inputs_[0].get_shape());
    // Directly add the tensor values
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = inputs_[0][i] + inputs_[1][i];
    }
    return result;
}
std::vector<Tensor> Add::backward(const Tensor& grad_output)
{
    return {grad_output,grad_output};
}


Tensor Multiply::forward(const std::vector<Tensor>& inputs_)
{
    inputs = inputs_;
    // Create result tensor with same shape
    Tensor result(inputs_[0].get_shape());
    // Directly multiply the tensor values
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = inputs_[0][i] * inputs_[1][i];
    }
    return result;
}
std::vector<Tensor> Multiply::backward(const Tensor& grad_output)
{
    const Tensor& x = inputs[0];
    const Tensor& y = inputs[1];
    
    if (x.size() == 1 && y.size() == 1) {
        // dz/dx = y * grad_output
        Tensor dx({y[0] * grad_output[0]}, false);
        // dz/dy = x * grad_output
        Tensor dy({x[0] * grad_output[0]}, false);
        return {dx, dy};
    }
    
    Tensor dx(x.get_shape());
    for (size_t i = 0; i < dx.size(); ++i) {
        dx[i] = y[i] * grad_output[i];
    }
    
    Tensor dy(y.get_shape());
    for (size_t i = 0; i < dy.size(); ++i) {
        dy[i] = x[i] * grad_output[i];
    }
    
    return {dx, dy};
}

Tensor Subtract::forward(const std::vector<Tensor>& inputs_)
{
    inputs = inputs_;
    Tensor result(inputs_[0].get_shape());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = inputs_[0][i] - inputs_[1][i];
    }
    return result;
}
std::vector<Tensor> Subtract::backward(const Tensor &grad_output)
{
    return {grad_output, grad_output * Tensor({-1.0f})}; // add -1 tp the y

}
Tensor Divide::forward(const std::vector<Tensor> &inputs_)
{
    inputs = inputs_;
    Tensor result(inputs_[0].get_shape());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = inputs_[0][i] / inputs_[1][i];
    }
    return result;
}
std::vector<Tensor> Divide::backward(const Tensor &grad_output)
{
    const Tensor& x = inputs[0];
    const Tensor& y = inputs[1];

    // dz/dx = 1/y
    // dz/dy = -x / y^2
    Tensor dx = grad_output / y;
    Tensor dy = grad_output * (x / (y * y)) * Tensor({-1.0f});
    return {dx,dy};

}