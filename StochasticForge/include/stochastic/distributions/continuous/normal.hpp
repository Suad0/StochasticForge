//
//  normal.hpp
//  StochasticForge
//
//  Created by Suad Demiri on 19.06.25.
//

// include/stochastic/distributions/continuous/normal.hpp
#pragma once
#include "../../core/distribution.hpp"
#include <cmath>

namespace stochastic {

template<typename T>
class Normal : public Distribution<T> {
private:
    T mu_, sigma_;
    mutable std::normal_distribution<T> dist_;
    
public:
    Normal(T mean = 0.0, T std_dev = 1.0)
    : mu_(mean), sigma_(std_dev), dist_(mean, std_dev) {
        if (sigma_ <= 0) {
            throw std::invalid_argument("Standard deviation must be positive");
        }
    }
    
    T sample() const override {
        return dist_(this->rng_);
    }
    
    T mean() const override { return mu_; }
    T variance() const override { return sigma_ * sigma_; }
    
    T pdf(const T& x) const override {
        const T z = (x - mu_) / sigma_;
        return std::exp(-0.5 * z * z) / (sigma_ * std::sqrt(2 * M_PI));
    }
    
    T cdf(const T& x) const override {
        const T z = (x - mu_) / (sigma_ * std::sqrt(2));
        return 0.5 * (1 + std::erf(z));
    }
    
protected:
    T third_central_moment() const override { return 0; } // Normal is symmetric
    T fourth_central_moment() const override { return 3 * std::pow(sigma_, 4); }
    
public:
    // Normal-specific methods
    T quantile(T p) const {
        if (p <= 0 || p >= 1) {
            throw std::invalid_argument("Quantile must be in (0, 1)");
        }
        // Approximation using inverse error function
        return mu_ + sigma_ * std::sqrt(2) * inverse_erf(2 * p - 1);
    }
    
private:
    // Simple approximation for inverse error function
    T inverse_erf(T x) const {
        const T a = 0.147;
        const T term1 = (2.0 / (M_PI * a)) + std::log(1 - x * x) / 2.0;
        const T term2 = std::log(1 - x * x) / a;
        return std::copysign(std::sqrt(-term1 + std::sqrt(term1 * term1 - term2)), x);
    }
};

} // namespace stochastic
