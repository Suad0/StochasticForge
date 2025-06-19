//
//  sde.hpp
//  StochasticForge
//
//  Created by Suad Demiri on 19.06.25.
//

// include/stochastic/processes/sde.hpp
#pragma once
#include <functional>
#include <vector>

namespace stochastic {

template<typename T>
class SDE {
public:
    // drift coefficient μ(t, X_t)
    virtual T drift(T t, T x) const = 0;
    
    // diffusion coefficient σ(t, X_t)
    virtual T diffusion(T t, T x) const = 0;
    
    virtual ~SDE() = default;
    
    // Optional: derivatives for higher-order methods
    virtual T drift_derivative_x(T t, T x) const { return 0; }
    virtual T diffusion_derivative_x(T t, T x) const { return 0; }
};

// Geometric Brownian Motion: dX = μX dt + σX dW
template<typename T>
class GeometricBrownianMotion : public SDE<T> {
private:
    T mu_, sigma_;
    
public:
    GeometricBrownianMotion(T drift, T volatility) : mu_(drift), sigma_(volatility) {}
    
    T drift(T t, T x) const override { return mu_ * x; }
    T diffusion(T t, T x) const override { return sigma_ * x; }
    
    T drift_derivative_x(T t, T x) const override { return mu_; }
    T diffusion_derivative_x(T t, T x) const override { return sigma_; }
};

// Ornstein-Uhlenbeck Process: dX = θ(μ - X) dt + σ dW
template<typename T>
class OrnsteinUhlenbeck : public SDE<T> {
private:
    T theta_, mu_, sigma_;
    
public:
    OrnsteinUhlenbeck(T mean_reversion_speed, T long_term_mean, T volatility)
    : theta_(mean_reversion_speed), mu_(long_term_mean), sigma_(volatility) {}
    
    T drift(T t, T x) const override { return theta_ * (mu_ - x); }
    T diffusion(T t, T x) const override { return sigma_; }
    
    T drift_derivative_x(T t, T x) const override { return -theta_; }
    T diffusion_derivative_x(T t, T x) const override { return 0; }
};

// Cox-Ingersoll-Ross Model: dX = κ(θ - X) dt + σ√X dW
template<typename T>
class CoxIngersollRoss : public SDE<T> {
private:
    T kappa_, theta_, sigma_;
    
public:
    CoxIngersollRoss(T mean_reversion, T long_term_mean, T volatility)
    : kappa_(mean_reversion), theta_(long_term_mean), sigma_(volatility) {}
    
    T drift(T t, T x) const override { return kappa_ * (theta_ - x); }
    T diffusion(T t, T x) const override { return sigma_ * std::sqrt(std::max(x, T(0))); }
    
    T drift_derivative_x(T t, T x) const override { return -kappa_; }
    T diffusion_derivative_x(T t, T x) const override {
        return x > 0 ? sigma_ / (2 * std::sqrt(x)) : 0;
    }
};

} // namespace stochastic

// include/stochastic/solvers/euler_maruyama.hpp
#pragma once
#include "../processes/sde.hpp"
#include <random>
#include <cmath>

namespace stochastic {

template<typename T>
class EulerMaruyama {
private:
    mutable std::mt19937 rng_;
    mutable std::normal_distribution<T> normal_dist_;
    
public:
    explicit EulerMaruyama(unsigned seed = std::random_device{}())
    : rng_(seed), normal_dist_(0.0, 1.0) {}
    
    std::vector<T> solve(const SDE<T>& sde, T x0, T t_start, T t_end, size_t n_steps) const {
        if (t_end <= t_start || n_steps == 0) {
            throw std::invalid_argument("Invalid time range or step count");
        }
        
        T dt = (t_end - t_start) / n_steps;
        std::vector<T> path;
        path.reserve(n_steps + 1);
        path.push_back(x0);
        
        T x = x0;
        T t = t_start;
        
        for (size_t i = 0; i < n_steps; ++i) {
            T dW = std::sqrt(dt) * normal_dist_(rng_);
            
            // Euler-Maruyama step: X_{n+1} = X_n + μ(t_n, X_n)Δt + σ(t_n, X_n)ΔW_n
            x += sde.drift(t, x) * dt + sde.diffusion(t, x) * dW;
            t += dt;
            
            path.push_back(x);
        }
        
        return path;
    }
    
    // Multiple paths simulation
    std::vector<std::vector<T>> solve_multiple(const SDE<T>& sde, T x0, T t_start, T t_end,
                                               size_t n_steps, size_t n_paths) const {
        std::vector<std::vector<T>> paths;
        paths.reserve(n_paths);
        
        for (size_t i = 0; i < n_paths; ++i) {
            paths.push_back(solve(sde, x0, t_start, t_end, n_steps));
        }
        
        return paths;
    }
    
    void set_seed(unsigned seed) { rng_.seed(seed); }
};

} // namespace stochastic

// include/stochastic/solvers/milstein.hpp
#pragma once
#include "../processes/sde.hpp"
#include <random>
#include <cmath>

namespace stochastic {

template<typename T>
class Milstein {
private:
    mutable std::mt19937 rng_;
    mutable std::normal_distribution<T> normal_dist_;
    
public:
    explicit Milstein(unsigned seed = std::random_device{}())
    : rng_(seed), normal_dist_(0.0, 1.0) {}
    
    std::vector<T> solve(const SDE<T>& sde, T x0, T t_start, T t_end, size_t n_steps) const {
        if (t_end <= t_start || n_steps == 0) {
            throw std::invalid_argument("Invalid time range or step count");
        }
        
        T dt = (t_end - t_start) / n_steps;
        std::vector<T> path;
        path.reserve(n_steps + 1);
        path.push_back(x0);
        
        T x = x0;
        T t = t_start;
        
        for (size_t i = 0; i < n_steps; ++i) {
            T dW = std::sqrt(dt) * normal_dist_(rng_);
            
            // Milstein method includes correction term
            T drift_term = sde.drift(t, x) * dt;
            T diffusion_term = sde.diffusion(t, x) * dW;
            T correction_term = 0.5 * sde.diffusion(t, x) * sde.diffusion_derivative_x(t, x) *
            (dW * dW - dt);
            
            x += drift_term + diffusion_term + correction_term;
            t += dt;
            
            path.push_back(x);
        }
        
        return path;
    }
    
    std::vector<std::vector<T>> solve_multiple(const SDE<T>& sde, T x0, T t_start, T t_end,
                                               size_t n_steps, size_t n_paths) const {
        std::vector<std::vector<T>> paths;
        paths.reserve(n_paths);
        
        for (size_t i = 0; i < n_paths; ++i) {
            paths.push_back(solve(sde, x0, t_start, t_end, n_steps));
        }
        
        return paths;
    }
    
    void set_seed(unsigned seed) { rng_.seed(seed); }
};

} // namespace stochastic

// include/stochastic/processes/poisson_process.hpp
#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

namespace stochastic {

template<typename T>
class PoissonProcess {
private:
    T lambda_;  // intensity rate
    T t_start_, t_end_;
    mutable std::mt19937 rng_;
    mutable std::exponential_distribution<T> exp_dist_;
    mutable std::poisson_distribution<int> poisson_dist_;
    
public:
    PoissonProcess(T intensity, T start_time = 0, T end_time = 1,
                   unsigned seed = std::random_device{}())
    : lambda_(intensity), t_start_(start_time), t_end_(end_time),
    rng_(seed), exp_dist_(intensity) {
        
        if (intensity <= 0) {
            throw std::invalid_argument("Intensity must be positive");
        }
        if (end_time <= start_time) {
            throw std::invalid_argument("End time must be greater than start time");
        }
    }
    
    // Generate event times using inter-arrival times
    std::vector<T> generate_event_times() const {
        std::vector<T> event_times;
        T current_time = t_start_;
        
        while (current_time < t_end_) {
            T inter_arrival = exp_dist_(rng_);
            current_time += inter_arrival;
            
            if (current_time < t_end_) {
                event_times.push_back(current_time);
            }
        }
        
        return event_times;
    }
    
    // Generate counting process N(t)
    std::vector<std::pair<T, int>> generate_counting_process(size_t n_points = 1000) const {
        auto event_times = generate_event_times();
        std::vector<std::pair<T, int>> counting_process;
        
        T dt = (t_end_ - t_start_) / n_points;
        int count = 0;
        size_t event_index = 0;
        
        for (size_t i = 0; i <= n_points; ++i) {
            T t = t_start_ + i * dt;
            
            // Count events up to time t
            while (event_index < event_times.size() && event_times[event_index] <= t) {
                count++;
                event_index++;
            }
            
            counting_process.emplace_back(t, count);
        }
        
        return counting_process;
    }
    
    // Generate number of events in time interval [0, t]
    int generate_count(T time_interval) const {
        if (time_interval <= 0) return 0;
        
        // Use Poisson distribution with parameter λt
        std::poisson_distribution<int> dist(lambda_ * time_interval);
        return dist(rng_);
    }
    
    // Generate multiple independent realizations
    std::vector<std::vector<T>> generate_multiple_realizations(size_t n_realizations) const {
        std::vector<std::vector<T>> realizations;
        realizations.reserve(n_realizations);
        
        for (size_t i = 0; i < n_realizations; ++i) {
            realizations.push_back(generate_event_times());
        }
        
        return realizations;
    }
    
    // Theoretical properties
    T theoretical_mean_count(T time_interval) const {
        return lambda_ * std::max(T(0), time_interval);
    }
    
    T theoretical_variance_count(T time_interval) const {
        return lambda_ * std::max(T(0), time_interval);
    }
    
    // Compound Poisson process
    template<typename Distribution>
    std::vector<std::pair<T, T>> generate_compound_process(const Distribution& jump_dist) const {
        auto event_times = generate_event_times();
        std::vector<std::pair<T, T>> compound_process;
        
        T cumulative_sum = 0;
        for (const auto& time : event_times) {
            T jump_size = jump_dist.sample();
            cumulative_sum += jump_size;
            compound_process.emplace_back(time, cumulative_sum);
        }
        
        return compound_process;
    }
    
    // Getters
    T get_intensity() const { return lambda_; }
    T get_start_time() const { return t_start_; }
    T get_end_time() const { return t_end_; }
    
    void set_seed(unsigned seed) {
        rng_.seed(seed);
        exp_dist_.reset();
    }
};

} // namespace stochastic
