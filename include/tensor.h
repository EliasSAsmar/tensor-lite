/**
 * @file tensor.h
 * @author Eli Asmar
 *
 *
 */

#ifndef TENSOR_LITE__TENSOR_H
#define TENSOR_LITE__TENSOR_H

#include <vector>
#include <cstddef>  // for size_t
#include <initializer_list>
#include <iostream>

class Tensor {
public:
    // === Constructors ===

    // Constructor: Create a tensor with a given shape (e.g., Tensor({2, 3}))
    Tensor(const std::vector<size_t>& shape_)
        : shape(shape_)
    {
        size_t total_size = 1;
        for (size_t dim : shape)
            total_size *= dim;

        data.resize(total_size, 0.0f);

        compute_strides();

    }

    // Constructor: 1D tensor from initializer list (e.g., Tensor{1.0, 2.0})
    Tensor(std::initializer_list<float> values)
        : data(values), shape{values.size()} {
        compute_strides();
    }


    // Flat indexing (e.g., tensor[0]) → returns float&
    float& operator[](size_t idx){return data[idx];}

    // Const version of flat indexing
    const float& operator[](size_t idx) const{return data[idx];}

    // Multi-index access (e.g., tensor.at({0, 1})) → returns float&
    float& at(const std::vector<size_t>& indices){return data[offset(indices)];}

    // Const version of multi-index access
    const float& at(const std::vector<size_t>& indices) const{return data[offset(indices)];}


    // === Arithmetic Operations ===

    // Element-wise addition (returns a new Tensor)
    Tensor operator+(const Tensor& other) const;

    // Element-wise multiplication (returns a new Tensor)
    Tensor operator*(const Tensor& other) const;

    // (Optional) Element-wise subtraction
    Tensor operator-(const Tensor& other) const;

    // (Optional) Element-wise division
    Tensor operator/(const Tensor& other) const;


    // === Utility ===

    // Print data as flat array (just for debugging)
    void print_flat() const;

    // Print shape of tensor
    void print_shape() const;

    // === Accessors ===
    const std::vector<float>& get_data() const { return data; }
    const std::vector<size_t>& get_shape() const { return shape; }
    const std::vector<size_t>& get_strides() const { return strides; }

// Optional: single value
    size_t size() const { return data.size(); }

private:
    // === Helpers ===

    // Calculate strides based on shape
    void compute_strides();

    // Convert multi-index to flat index using strides
    size_t offset(const std::vector<size_t>& indices) const;

    // Ensure two tensors have the same shape
    void check_shape_match(const Tensor& other) const;

    // Data members
    std::vector<float> data;          // Flat storage of values
    std::vector<size_t> shape;        // Shape of tensor (e.g., {2, 3} for 2x3)
    std::vector<size_t> strides;      // Strides to map multi-index → flat index
};

#endif //TENSOR_LITE__TENSOR_H
