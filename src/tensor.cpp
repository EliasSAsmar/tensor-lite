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
    Tensor result(shape);

    for (size_t i =0; i < data.size(); ++i)
        result[i] = data[i] * other[i];
    return result;
}

Tensor Tensor::operator+(const Tensor &other) const
{
    Tensor::check_shape_match(other);
    Tensor result(shape);

    for (size_t i =0; i < data.size(); ++i)
        result[i] = data[i] + other[i];
    return result;

}

Tensor Tensor::operator/(const Tensor &other) const
{
    Tensor::check_shape_match(other);
    Tensor result(shape);

    for (size_t i =0; i < data.size(); ++i)
        result[i] = data[i] / other[i];
    return result;

}

Tensor Tensor::operator-(const Tensor &other) const
{
    Tensor::check_shape_match(other);
    Tensor result(shape);

    for (size_t i =0; i < data.size(); ++i)
        result[i] = data[i] - other[i];
    return result;

}

