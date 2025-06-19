# 🧪 StochasticForge

**StochasticForge** is a modern C++ library for stochastic analysis, probability theory, and simulation of stochastic processes.  
Built from scratch as a learning and research tool, it provides performant building blocks for working with probability distributions, stochastic calculus, and stochastic differential equations (SDEs).

---

## 🚀 Features

- 🎲 **Probability Distributions**: Normal, Bernoulli, Uniform, and more
- 📊 **Random Variables**: Abstract base class for probabilistic models
- 📈 **Stochastic Processes**: Brownian motion, Markov chains, SDE models
- 🔬 **Stochastic Calculus**: Ito integrals and Ito's Lemma
- ⚙️ **Numerical Solvers**: Euler-Maruyama and Milstein methods
- 🔁 **Monte Carlo Simulation**: Sampling-based estimation framework
- 🧠 **Statistics**: Mean, variance, covariance, and correlation

---

## 📦 Project Structure

```bash
StochasticForge/
├── include/stochastic/      # Public headers
│   ├── core/                # Core classes (random variables, samplers, etc.)
│   ├── distributions/       # Discrete & continuous probability distributions
│   ├── processes/           # Stochastic processes like Brownian motion
│   ├── calculus/            # Ito integrals and related tools
│   └── solvers/             # Numerical SDE solvers
├── src/                     # Implementation source files
├── tests/                   # Unit tests
├── examples/                # Example programs and simulations
├── main.cpp                 # Optional entry point
└── CMakeLists.txt           # Build system configuration
```

---

## 🛠️ Build Instructions

### 🔧 Requirements

- C++20 compatible compiler (Clang, GCC, or MSVC)
- CMake ≥ 3.16
- (Optional) [Catch2](https://github.com/catchorg/Catch2) for unit tests
- (Optional) Xcode or CLion for IDE support

### 🧱 Building from Source

```bash
git clone https://github.com/yourusername/StochasticForge.git
cd StochasticForge
mkdir build && cd build
cmake ..
make
./StochasticForge
```

---

## 📚 Example: Sampling from a Normal Distribution

```cpp
#include "stochastic/distributions/continuous/normal.hpp"
#include <iostream>

int main() {
    stochastic::NormalDistribution norm(0.0, 1.0);
    std::cout << "Sample: " << norm.sample() << std::endl;
}
```

---

## 🔭 Roadmap

- [x] Base distribution & random variable classes
- [x] Continuous and discrete distributions
- [ ] Monte Carlo simulation engine
- [ ] Brownian motion & Ito calculus support
- [ ] Euler-Maruyama and Milstein SDE solvers
- [ ] Financial modeling (e.g. Black-Scholes)
- [ ] Interactive demos or Python bindings

---

## 🤝 Contributing

This project is a personal deep-dive into stochastic mathematics and C++ systems development, but contributions are welcome.

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/AmazingThing`
3. Commit your changes: `git commit -am 'Add some AmazingThing'`
4. Push to the branch: `git push origin feature/AmazingThing`
5. Open a pull request

---



## 👋 Author

Created by **Suad** — exploring stochastic processes, one line of C++ at a time.  
Feel free to connect or contribute ideas!

---

## ⭐️ Show Support

If you find this library helpful or educational, feel free to star the project and share your feedback!

