
# 🧠 LightTensor

A lightweight machine learning framework written in **modern C++**, featuring:

- N-dimensional tensor support
- Automatic differentiation (autograd)
- Neural network layers
- Optimizers like SGD and Adam
- Model training interface

This project is built from scratch to better understand the internals of deep learning frameworks like PyTorch and TensorFlow — with a strong focus on performance and modularity.

---

## 🚀 Features (WIP)

- [x] Custom `Tensor` class with basic arithmetic and shape manipulation  
- [x] Broadcasting and slicing  
- [ ] Computation graph for autograd  
- [ ] Backpropagation for gradient computation  
- [ ] Neural network layers (Linear, ReLU, etc.)  
- [ ] Loss functions (MSE, Cross-Entropy)  
- [ ] Optimizers (SGD, Adam)  
- [ ] Dataset loaders and training loop  
- [ ] CUDA acceleration (planned)  

---

## 📁 Project Structure

```
LightTensor/
├── include/        # Header files (Tensor, Autograd, Layers, etc.)
├── src/            # Source files
├── tests/          # Unit tests for each module
├── examples/       # Training examples (linear regression, MLP)
├── CMakeLists.txt  # Build configuration
└── README.md       # This file
```

---

## 🧪 Build & Run

Make sure you have CMake and a modern C++ compiler (C++17 or higher):

```bash
git clone https://github.com/yourusername/LightTensor.git
cd LightTensor
mkdir build && cd build
cmake ..
make
./run_tests
```

---

## 📦 Dependencies

- C++17 or higher
- CMake 3.15+
- (Optional) [Catch2](https://github.com/catchorg/Catch2) or GoogleTest for unit testing
- (Planned) CUDA Toolkit for GPU support

---

## 🧠 Motivation

This project was created as a grind challenge to:
- Improve low-level C++ skills (memory management, performance tuning)
- Understand how real ML frameworks like PyTorch and TensorFlow work internally
- Build something difficult and educational from scratch, no dependencies, no training wheels

---

## ✍️ Author

**Elias Sammy Asmar**  
Computer Science @ MSU  
[GitHub](https://github.com/yourusername) • [LinkedIn](https://linkedin.com/in/yourprofile)

---

## 📜 License

This project is licensed under the MIT License.
```

