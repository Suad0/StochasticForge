//
//  sampler.hpp
//  StochasticForge
//
//  Created by Suad Demiri on 19.06.25.
//

// include/stochastic/core/sampler.hpp
#pragma once
#include "random_variable.hpp"
#include <memory>
#include <functional>

namespace stochastic {

template<typename T>
class MonteCarloSampler {
private:
    std::unique_ptr<RandomVariable<T>> rv_;
    size_t n_samples_;
    
public:
    MonteCarloSampler(std::unique_ptr<RandomVariable<T>> rv, size_t n)
    : rv_(std::move(rv)), n_samples_(n) {}
    
    std::vector<T> sample_path() const {
        std::vector<T> samples;
        samples.reserve(n_samples_);
        for (size_t i = 0; i < n_samples_; ++i) {
            samples.push_back(rv_->sample());
        }
        return samples;
    }
    
    T estimate_expectation(std::function<T(const T&)> func) const {
        T sum = 0;
        for (size_t i = 0; i < n_samples_; ++i) {
            sum += func(rv_->sample());
        }
        return sum / n_samples_;
    }
    
    // Confidence interval for expectation estimation
    std::pair<T, T> expectation_confidence_interval(
                                                    std::function<T(const T&)> func,
                                                    T confidence_level = 0.95) const {
                                                        
                                                        std::vector<T> samples;
                                                        samples.reserve(n_samples_);
                                                        
                                                        for (size_t i = 0; i < n_samples_; ++i) {
                                                            samples.push_back(func(rv_->sample()));
                                                        }
                                                        
                                                        const T sample_mean = Statistics<T>::mean(samples);
                                                        const T sample_std = Statistics<T>::standard_deviation(samples);
                                                        const T z_score = 1.96; // Approximate for 95% confidence
                                                        const T margin = z_score * sample_std / std::sqrt(n_samples_);
                                                        
                                                        return {sample_mean - margin, sample_mean + margin};
                                                    }
    
    void set_sample_size(size_t n) { n_samples_ = n; }
    size_t get_sample_size() const { return n_samples_; }
};

} // namespace stochastic
