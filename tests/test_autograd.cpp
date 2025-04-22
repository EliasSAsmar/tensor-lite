/**
 * @file test_autograd.cpp
 * @author Eli Asmar
 */
#include <gtest/gtest.h>
#include "tensor.h"


TEST(AutogradTest, AddBackward) {
    Tensor x({1.0f}, true);
    Tensor y({2.0f}, true);
    Tensor z = x + y;

    z.backward();

    EXPECT_FLOAT_EQ(x.get_grad()[0], 1.0f);
    EXPECT_FLOAT_EQ(y.get_grad()[0], 1.0f);
}

TEST(AutogradTest, MultiplyBackward) {
    Tensor x({3.0f}, true);
    Tensor y({4.0f}, true);
    Tensor z = x * y;

    z.backward();

    EXPECT_FLOAT_EQ(x.get_grad()[0], 4.0f); // dz/dx = y
    EXPECT_FLOAT_EQ(y.get_grad()[0], 3.0f); // dz/dy = x
}

TEST(AutogradTest, SubtractBackward) {
    Tensor x({5.0f}, true);
    Tensor y({2.0f}, true);
    Tensor z = x - y;

    z.backward();

    EXPECT_FLOAT_EQ(x.get_grad()[0], 1.0f);    // dz/dx = 1
    EXPECT_FLOAT_EQ(y.get_grad()[0], -1.0f);   // dz/dy = -1
}

TEST(AutogradTest, DivideBackward) {
    Tensor x({8.0f}, true);
    Tensor y({2.0f}, true);
    Tensor z = x / y;

    z.backward();

    EXPECT_FLOAT_EQ(x.get_grad()[0], 0.5f);    // dz/dx = 1/y = 0.5
    EXPECT_FLOAT_EQ(y.get_grad()[0], -2.0f);   // dz/dy = -x/y^2 = -8/4
}

TEST(AutogradTest, ChainRuleTest) {
    Tensor x({2.0f}, true);
    Tensor y({3.0f}, true);

    // z = x * y + x
    // Explicitly create intermediate tensor to avoid dangling pointer issues
    // with the current raw pointer implementation in _inputs
    Tensor tmp = x * y;
    Tensor z = tmp + x;
    z.backward();

    // dz/d(tmp) = 1, d(tmp)/dx = y => contribution = y
    // dz/dx = 1 => contribution = 1
    // Total dz/dx = y + 1 = 3 + 1 = 4
    EXPECT_FLOAT_EQ(x.get_grad()[0], 4.0f);

    // dz/d(tmp) = 1, d(tmp)/dy = x => contribution = x
    // Total dz/dy = x = 2
    EXPECT_FLOAT_EQ(y.get_grad()[0], 2.0f);
}
