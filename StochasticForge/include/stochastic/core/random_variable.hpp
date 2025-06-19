//
//  random_variable.hpp
//  StochasticForge
//
//  Created by Suad Demiri on 19.06.25.
//

// include/stochastic/core/random_variable.hpp
#pragma once
#include <functional>
#include <random>

namespace stochastic {

template<typename T>
class RandomVariable {
public:
    virtual ~RandomVariable() = default;
    
    // Core interface
    virtual T sample() const = 0;
    virtual T mean() const = 0;
    virtual T variance() const = 0;
    virtual T pdf(const T& x) const = 0;
    virtual T cdf(const T& x) const = 0;
    
    // Derived properties
    T standard_deviation() const { return std::sqrt(variance()); }
    T skewness() const { return third_central_moment() / std::pow(standard_deviation(), 3); }
    T kurtosis() const { return fourth_central_moment() / std::pow(variance(), 2); }
    
    // Monte Carlo expectation
    T expectation(std::function<T(const T&)> func, size_t n_samples = 10000) const {
        T sum = 0;
        for (size_t i = 0; i < n_samples; ++i) {
            sum += func(sample());
        }
        return sum / n_samples;
    }

protected:
    virtual T third_central_moment() const = 0;
    virtual T fourth_central_moment() const = 0;
};

} // namespace stochastic
