//
//  distribution.hpp
//  StochasticForge
//
//  Created by Suad Demiri on 19.06.25.
//

// include/stochastic/core/distribution.hpp
#pragma once
#include "random_variable.hpp"
#include <random>

namespace stochastic {

template<typename T>
class Distribution : public RandomVariable<T> {
protected:
    mutable std::mt19937 rng_;
    
public:
    explicit Distribution(unsigned seed = std::random_device{}()) : rng_(seed) {}
    
    // Seed management
    void set_seed(unsigned seed) { rng_.seed(seed); }
    unsigned get_seed() const { return rng_(); }
    
    // Batch sampling
    std::vector<T> sample(size_t n) const {
        std::vector<T> samples;
        samples.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            samples.push_back(sample());
        }
        return samples;
    }
};

} // namespace stochastic
