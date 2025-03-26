/**
 * @file test_tensor.cpp
 * @author Eli Asmar
 */

#include <gtest/gtest.h>
#include "tensor.h"

TEST(TensorTest, ShapeAndSize) {
Tensor t(std::vector<size_t>{2, 3});
EXPECT_EQ(t.get_shape().size(), 2);
EXPECT_EQ(t.get_shape()[0], 2);
EXPECT_EQ(t.get_shape()[1], 3);
EXPECT_EQ(t.size(), 6);
}

TEST(TensorTest, InitializerListConstructor) {
Tensor t = {1.0f, 2.0f, 3.0f};
EXPECT_EQ(t.get_shape().size(), 1);
EXPECT_EQ(t.get_shape()[0], 3);
EXPECT_FLOAT_EQ(t[0], 1.0f);
EXPECT_FLOAT_EQ(t[2], 3.0f);
}

TEST(TensorTest, MultiIndexAccess) {
Tensor t(std::vector<size_t>{2, 2});
t[0] = 1.0f;
t[1] = 2.0f;
t[2] = 3.0f;
t[3] = 4.0f;

EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
EXPECT_FLOAT_EQ(t.at({1, 1}), 4.0f);
}

TEST(TensorTest, OutOfBoundsIndexThrows) {
Tensor t(std::vector<size_t>{2, 2});
EXPECT_THROW(t.at({2, 0}), std::out_of_range); // Add if you implement bounds checking
}

TEST(TensorTest, ShapeMismatchAdditionThrows) {
Tensor a({2, 2});
Tensor b({4});
EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(TensorTest, ElementwiseAdditionWorks) {
    Tensor a = {1.0f, 2.0f, 3.0f};
    Tensor b = {4.0f, 5.0f, 6.0f};
    Tensor c = a + b;
    EXPECT_FLOAT_EQ(c[0], 5.0f);
    EXPECT_FLOAT_EQ(c[1], 7.0f);
    EXPECT_FLOAT_EQ(c[2], 9.0f);
}

TEST(TensorTest, ElementwiseMultiplicationWorks) {
    Tensor a = {1.0f, 2.0f, 3.0f};
    Tensor b = {2.0f, 4.0f, 6.0f};
    Tensor c = a * b;
    EXPECT_FLOAT_EQ(c[0], 2.0f);
    EXPECT_FLOAT_EQ(c[1], 8.0f);
    EXPECT_FLOAT_EQ(c[2], 18.0f);
}

TEST(TensorTest, ElementwiseSubtractionWorks) {
    Tensor a = {5.0f, 7.0f, 9.0f};
    Tensor b = {2.0f, 3.0f, 4.0f};
    Tensor c = a - b;
    EXPECT_FLOAT_EQ(c[0], 3.0f);
    EXPECT_FLOAT_EQ(c[1], 4.0f);
    EXPECT_FLOAT_EQ(c[2], 5.0f);
}

TEST(TensorTest, ElementwiseDivisionWorks) {
    Tensor a = {8.0f, 6.0f, 4.0f};
    Tensor b = {2.0f, 3.0f, 2.0f};
    Tensor c = a / b;
    EXPECT_FLOAT_EQ(c[0], 4.0f);
    EXPECT_FLOAT_EQ(c[1], 2.0f);
    EXPECT_FLOAT_EQ(c[2], 2.0f);
}

