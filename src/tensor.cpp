/**
 * @file tensor.cpp
 * @author Eli Asmar
 */

#include "../include/tensor.h"

void Tensor::compute_strides() {
    strides.resize(shape.size());
    size_t stride = 1;

    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

size_t Tensor::offset(const std::vector<size_t>& indices) const {
    if (indices.size() != shape.size())
        throw std::invalid_argument("Invalid dimensions for indexing.");

    size_t off = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (indices[i] >= shape[i])
            throw std::out_of_range("Index out of bounds at dimension " + std::to_string(i));
        off += indices[i] * strides[i];
    }


    return off;
}

void Tensor::check_shape_match(const Tensor &other) const
{
    if (shape != other.shape)
        throw std::invalid_argument("Invalid dimensions for indexing.");
}

Tensor Tensor::operator*(const Tensor &other) const
{
    Tensor::check_shape_match(other);

    auto mul_fn = std::make_shared<Multiply>();
    mul_fn->inputs = {*this, other};
    Tensor out = mul_fn->forward(mul_fn->inputs);

    if (this->requires_grad || other.requires_grad) {
        out.requires_grad = true;
        out.grad_fn = mul_fn;
        out._inputs = {this, &other};
    }

    return out;
}

Tensor Tensor::operator+(const Tensor &other) const
{
    check_shape_match(other);

    auto add_fn = std::make_shared<Add>();
    Tensor out = add_fn->forward({*this, other});

    if (this->requires_grad || other.requires_grad) {
        out.requires_grad = true;
        out.grad_fn = add_fn;
        out._inputs = {this, &other};
    }

    return out;

}

Tensor Tensor::operator/(const Tensor &other) const
{
    check_shape_match(other);

    auto div_fn = std::make_shared<Divide>();
    Tensor out = div_fn->forward({*this, other});

    if (this->requires_grad || other.requires_grad) {
        out.requires_grad = true;
        out.grad_fn = div_fn;
        out._inputs = {this, &other};
    }

    return out;


}

Tensor Tensor::operator-(const Tensor &other) const
{
    check_shape_match(other);

    auto sub_fn = std::make_shared<Subtract>();
    Tensor out = sub_fn->forward({*this, other});

    if (this->requires_grad || other.requires_grad) {
        out.requires_grad = true;
        out.grad_fn = sub_fn;
        out._inputs = {this, &other};
    }

    return out;
}


// src/tensor.cpp

void Tensor::backward(const Tensor& grad_output) const
{
    if (!requires_grad) return;

    if (!grad) {
        grad = std::make_shared<Tensor>(shape);
        if (grad->data.size() != grad_output.data.size()) {
             if (grad_output.size() == 1) {
                 std::fill(grad->data.begin(), grad->data.end(), grad_output.data[0]);
             } else {
                 throw std::runtime_error("Gradient initialization shape mismatch.");
             }
        } else {
            std::copy(grad_output.data.begin(), grad_output.data.end(), grad->data.begin());
        }
    } else {
        if (grad->data.size() != grad_output.data.size()) {
            if (grad_output.size() == 1) {
                float grad_val = grad_output.data[0];
                for (size_t i = 0; i < grad->data.size(); ++i) {
                    grad->data[i] += grad_val;
                }
            } else {
                throw std::runtime_error("Gradient accumulation shape mismatch.");
            }
        } else {
            for (size_t i = 0; i < grad->data.size(); ++i) {
                grad->data[i] += grad_output.data[i];
            }
        }
    }


    if (grad_fn) {
        std::vector<Tensor> grad_inputs = grad_fn->backward(*grad);

        if (grad_inputs.size() != this->_inputs.size()) {
             throw std::runtime_error("Mismatch between number of gradients and inputs ptrs in backward pass.");
        }


        for (size_t i = 0; i < this->_inputs.size(); ++i) {
            if (!this->_inputs[i]) {
                 throw std::runtime_error("Null input tensor pointer in backward pass.");
            }
            this->_inputs[i]->backward(grad_inputs[i]);
        }
    }
}

void Tensor::backward() const {
    if (data.size() != 1)
        throw std::runtime_error("Only scalar outputs can initiate backward().");

    Tensor grad_output({1.0f});
    this->backward(grad_output);
}

