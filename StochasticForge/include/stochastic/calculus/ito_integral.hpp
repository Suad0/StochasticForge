//
//  ito_integral.hpp
//  StochasticForge
//
//  Created by Suad Demiri on 19.06.25.
//


// include/stochastic/calculus/ito_integral.hpp
#pragma once
#include <vector>
#include <functional>
#include <random>
#include <cmath>

namespace stochastic {

template<typename T>
class ItoIntegral {
private:
    mutable std::mt19937 rng_;
    mutable std::normal_distribution<T> normal_dist_;
    
public:
    explicit ItoIntegral(unsigned seed = std::random_device{}())
    : rng_(seed), normal_dist_(0.0, 1.0) {}
    
    // Compute ∫f(t,W_t) dW_t using Riemann-Stieltjes approximation
    T compute_integral(std::function<T(T, T)> integrand,
                       const std::vector<T>& brownian_path,
                       const std::vector<T>& time_grid) const {
        
        if (brownian_path.size() != time_grid.size()) {
            throw std::invalid_argument("Brownian path and time grid must have same size");
        }
        
        if (brownian_path.size() < 2) {
            return T(0);
        }
        
        T integral = T(0);
        for (size_t i = 0; i < brownian_path.size() - 1; ++i) {
            T t = time_grid[i];
            T W_t = brownian_path[i];
            T dW = brownian_path[i + 1] - brownian_path[i];
            
            integral += integrand(t, W_t) * dW;
        }
        
        return integral;
    }
    
    // Compute ∫f(t) dW_t where f is deterministic
    T compute_deterministic_integral(std::function<T(T)> f,
                                     const std::vector<T>& brownian_path,
                                     const std::vector<T>& time_grid) const {
        
        return compute_integral([f](T t, T W) { return f(t); }, brownian_path, time_grid);
    }
    
    // Compute ∫W_t dW_t = (W_T^2 - T)/2 (exact formula)
    T compute_W_dW_integral(const std::vector<T>& brownian_path,
                            const std::vector<T>& time_grid) const {
        
        if (brownian_path.empty() || time_grid.empty()) return T(0);
        
        T W_T = brownian_path.back();
        T T = time_grid.back() - time_grid.front();
        
        return (W_T * W_T - T) / 2;
    }
    
    // Monte Carlo estimation of Ito integral
    T monte_carlo_integral(std::function<T(T, T)> integrand,
                           T t_start, T t_end, size_t n_steps,
                           size_t n_simulations = 1000) const {
        
        T sum = T(0);
        T dt = (t_end - t_start) / n_steps;
        
        for (size_t sim = 0; sim < n_simulations; ++sim) {
            // Generate Brownian path
            std::vector<T> W_path;
            std::vector<T> time_grid;
            W_path.reserve(n_steps + 1);
            time_grid.reserve(n_steps + 1);
            
            W_path.push_back(T(0));
            time_grid.push_back(t_start);
            
            for (size_t i = 0; i < n_steps; ++i) {
                T dW = std::sqrt(dt) * normal_dist_(rng_);
                W_path.push_back(W_path.back() + dW);
                time_grid.push_back(t_start + (i + 1) * dt);
            }
            
            sum += compute_integral(integrand, W_path, time_grid);
        }
        
        return sum / n_simulations;
    }
    
    // Stratonovich integral (for comparison)
    T compute_stratonovich_integral(std::function<T(T, T)> integrand,
                                    const std::vector<T>& brownian_path,
                                    const std::vector<T>& time_grid) const {
        
        if (brownian_path.size() != time_grid.size() || brownian_path.size() < 2) {
            return T(0);
        }
        
        T integral = T(0);
        for (size_t i = 0; i < brownian_path.size() - 1; ++i) {
            T t_i = time_grid[i];
            T t_ip1 = time_grid[i + 1];
            T W_i = brownian_path[i];
            T W_ip1 = brownian_path[i + 1];
            T dW = W_ip1 - W_i;
            
            // Stratonovich uses midpoint rule
            T t_mid = (t_i + t_ip1) / 2;
            T W_mid = (W_i + W_ip1) / 2;
            
            integral += integrand(t_mid, W_mid) * dW;
        }
        
        return integral;
    }
    
    void set_seed(unsigned seed) { rng_.seed(seed); }
};

} // namespace stochastic


