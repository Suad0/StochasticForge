//
//  brownian_motion.hpp
//  StochasticForge
//
//  Created by Suad Demiri on 19.06.25.
//

// include/stochastic/processes/brownian_motion.hpp
#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>

namespace stochastic {

template<typename T>
class BrownianMotion {
private:
    T t_start_, t_end_, dt_;
    size_t n_steps_;
    T mu_, sigma_;  // drift and volatility
    T initial_value_;
    mutable std::mt19937 rng_;
    mutable std::normal_distribution<T> normal_dist_;
    
public:
    BrownianMotion(T start_time, T end_time, size_t steps,
                   T drift = 0.0, T volatility = 1.0, T initial = 0.0,
                   unsigned seed = std::random_device{}())
    : t_start_(start_time), t_end_(end_time), n_steps_(steps),
    mu_(drift), sigma_(volatility), initial_value_(initial),
    rng_(seed), normal_dist_(0.0, 1.0) {
        
        if (end_time <= start_time) {
            throw std::invalid_argument("End time must be greater than start time");
        }
        if (steps == 0) {
            throw std::invalid_argument("Number of steps must be positive");
        }
        if (volatility < 0) {
            throw std::invalid_argument("Volatility must be non-negative");
        }
        
        dt_ = (t_end_ - t_start_) / steps;
    }
    
    // Generate a complete path
    std::vector<T> generate_path() const {
        std::vector<T> path;
        path.reserve(n_steps_ + 1);
        path.push_back(initial_value_);
        
        T current_value = initial_value_;
        for (size_t i = 0; i < n_steps_; ++i) {
            T dW = std::sqrt(dt_) * normal_dist_(rng_);
            current_value += mu_ * dt_ + sigma_ * dW;
            path.push_back(current_value);
        }
        
        return path;
    }
    
    // Generate Brownian increments dW_t
    std::vector<T> generate_increments() const {
        std::vector<T> increments;
        increments.reserve(n_steps_);
        
        for (size_t i = 0; i < n_steps_; ++i) {
            increments.push_back(std::sqrt(dt_) * normal_dist_(rng_));
        }
        
        return increments;
    }
    
    // Generate multiple independent paths
    std::vector<std::vector<T>> generate_paths(size_t n_paths) const {
        std::vector<std::vector<T>> paths;
        paths.reserve(n_paths);
        
        for (size_t i = 0; i < n_paths; ++i) {
            paths.push_back(generate_path());
        }
        
        return paths;
    }
    
    // Get value at specific time (interpolated if necessary)
    T get_value_at(T time, const std::vector<T>& path) const {
        if (time < t_start_ || time > t_end_) {
            throw std::out_of_range("Time is outside the simulation range");
        }
        
        if (path.size() != n_steps_ + 1) {
            throw std::invalid_argument("Path size doesn't match expected size");
        }
        
        T relative_time = (time - t_start_) / (t_end_ - t_start_);
        T exact_index = relative_time * n_steps_;
        
        size_t lower_index = static_cast<size_t>(std::floor(exact_index));
        if (lower_index >= n_steps_) {
            return path.back();
        }
        
        // Linear interpolation
        T weight = exact_index - lower_index;
        return path[lower_index] * (1 - weight) + path[lower_index + 1] * weight;
    }
    
    // Statistical properties
    T theoretical_mean(T time) const {
        if (time < t_start_) return initial_value_;
        return initial_value_ + mu_ * (time - t_start_);
    }
    
    T theoretical_variance(T time) const {
        if (time < t_start_) return 0;
        return sigma_ * sigma_ * (time - t_start_);
    }
    
    T theoretical_std(T time) const {
        return std::sqrt(theoretical_variance(time));
    }
    
    // Covariance between two time points
    T theoretical_covariance(T t1, T t2) const {
        if (t1 < t_start_ || t2 < t_start_) return 0;
        T s = std::min(t1, t2) - t_start_;
        return sigma_ * sigma_ * s;
    }
    
    // Generate correlated Brownian motions
    std::pair<std::vector<T>, std::vector<T>> generate_correlated_paths(T correlation) const {
        if (correlation < -1 || correlation > 1) {
            throw std::invalid_argument("Correlation must be in [-1, 1]");
        }
        
        std::vector<T> path1, path2;
        path1.reserve(n_steps_ + 1);
        path2.reserve(n_steps_ + 1);
        
        path1.push_back(initial_value_);
        path2.push_back(initial_value_);
        
        T current_value1 = initial_value_;
        T current_value2 = initial_value_;
        
        for (size_t i = 0; i < n_steps_; ++i) {
            T Z1 = normal_dist_(rng_);
            T Z2 = normal_dist_(rng_);
            
            T dW1 = std::sqrt(dt_) * Z1;
            T dW2 = std::sqrt(dt_) * (correlation * Z1 + std::sqrt(1 - correlation * correlation) * Z2);
            
            current_value1 += mu_ * dt_ + sigma_ * dW1;
            current_value2 += mu_ * dt_ + sigma_ * dW2;
            
            path1.push_back(current_value1);
            path2.push_back(current_value2);
        }
        
        return {path1, path2};
    }
    
    // Geometric Brownian Motion (special case)
    std::vector<T> generate_geometric_path() const {
        std::vector<T> path;
        path.reserve(n_steps_ + 1);
        path.push_back(initial_value_);
        
        T current_value = initial_value_;
        for (size_t i = 0; i < n_steps_; ++i) {
            T dW = std::sqrt(dt_) * normal_dist_(rng_);
            // dS = S * (μ dt + σ dW)
            current_value *= (1 + mu_ * dt_ + sigma_ * dW);
            path.push_back(current_value);
        }
        
        return path;
    }
    
    // Bridge sampling: generate path conditioned on endpoint
    std::vector<T> generate_bridge_path(T end_value) const {
        std::vector<T> path;
        path.reserve(n_steps_ + 1);
        path.push_back(initial_value_);
        
        // Generate standard Brownian bridge and then transform
        std::vector<T> times;
        times.reserve(n_steps_ + 1);
        for (size_t i = 0; i <= n_steps_; ++i) {
            times.push_back(t_start_ + i * dt_);
        }
        
        T current_value = initial_value_;
        for (size_t i = 1; i < n_steps_; ++i) {
            T t = times[i];
            T remaining_time = t_end_ - t;
            T elapsed_time = t - t_start_;
            
            // Conditional mean and variance for Brownian bridge
            T bridge_mean = current_value + (end_value - current_value) * (dt_ / remaining_time);
            T bridge_var = sigma_ * sigma_ * dt_ * (remaining_time - dt_) / remaining_time;
            
            T increment = normal_dist_(rng_) * std::sqrt(bridge_var) +
            (bridge_mean - current_value) + mu_ * dt_;
            
            current_value += increment;
            path.push_back(current_value);
        }
        
        path.push_back(end_value);
        return path;
    }
    
    // Getters
    T get_dt() const { return dt_; }
    size_t get_steps() const { return n_steps_; }
    T get_drift() const { return mu_; }
    T get_volatility() const { return sigma_; }
    T get_start_time() const { return t_start_; }
    T get_end_time() const { return t_end_; }
    T get_initial_value() const { return initial_value_; }
    
    // Time vector for plotting
    std::vector<T> get_time_grid() const {
        std::vector<T> times;
        times.reserve(n_steps_ + 1);
        for (size_t i = 0; i <= n_steps_; ++i) {
            times.push_back(t_start_ + i * dt_);
        }
        return times;
    }
};

} // namespace stochastic
