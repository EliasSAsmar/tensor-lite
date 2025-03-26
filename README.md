```markdown
# LightTensor

LightTensor is a lightweight C++ tensor library built from scratch. It supports dynamic N-dimensional tensors, multi-index access, element-wise operations, and basic safety features. This forms the foundation for a minimal machine learning framework with future support for automatic differentiation and neural networks.

## Features

- N-dimensional tensor support
- Element-wise arithmetic: `+`, `-`, `*`, `/`
- Multi-index access using shape-aware offset computation
- Flat index access for internal operations
- Shape and stride computation for memory layout
- Bounds checking and shape validation
- Full test coverage using GoogleTest

## Project Structure

```
tensor-lite/
├── include/           # Public header files
│   └── tensor.h
├── src/               # Tensor implementation
│   └── tensor.cpp
├── tests/             # GoogleTest-based unit tests
│   └── test_tensor.cpp
├── CMakeLists.txt     # Root CMake file
└── README.md
```

## Build and Run Tests

Ensure CMake (3.15+) and a C++17-compatible compiler are installed.

```bash
mkdir build && cd build
cmake ..
make
ctest --output-on-failure
```

## Example Usage

```cpp
#include "tensor.h"

int main() {
    Tensor t(std::vector<size_t>{2, 3});
    t.at({1, 2}) = 5.0f;
    t.print_flat();     // [ 0.0 0.0 0.0 0.0 0.0 5.0 ]
    t.print_shape();    // Shape: (2, 3)
}
```

## Tests

The project uses GoogleTest for unit testing. Tests cover:

- Tensor construction (shape and value-based)
- Indexing and stride logic
- Arithmetic operations
- Shape mismatch detection
- Out-of-bounds access

Run with:

```bash
./run_tests
```

