//
//  ito_lemma.hpp
//  StochasticForge
//
//  Created by Suad Demiri on 19.06.25.
//

// include/stochastic/calculus/ito_lemma.hpp
#pragma once
#include <functional>
#include <vector>
#include <cmath>

namespace stochastic {

template<typename T>
class ItoLemma {
public:
    // Apply Ito's lemma to function f(t, X_t) where dX_t follows SDE
    // df = (∂f/∂t + μ∂f/∂x + 0.5σ²∂²f/∂x²)dt + σ∂f/∂x dW_t
    struct ItoResult {
        std::function<T(T, T)> drift_coeff;      // coefficient of dt
        std::function<T(T, T)> diffusion_coeff;  // coefficient of dW_t
    };
    
    // Generic Ito's lemma application
    static ItoResult apply_ito_lemma(
                                     std::function<T(T, T)> f,           // f(t, x)
                                     std::function<T(T, T)> df_dt,       // ∂f/∂t
                                     std::function<T(T, T)> df_dx,       // ∂f/∂x
                                     std::function<T(T, T)> d2f_dx2,     // ∂²f/∂x²
                                     std::function<T(T, T)> mu,          // drift of X_t
                                     std::function<T(T, T)> sigma        // diffusion of X_t
    ) {
        
        ItoResult result;
        
        // Drift coefficient: ∂f/∂t + μ∂f/∂x + 0.5σ²∂²f/∂x²
        result.drift_coeff = [df_dt, df_dx, d2f_dx2, mu, sigma](T t, T x) -> T {
            return df_dt(t, x) + mu(t, x) * df_dx(t, x) +
            0.5 * sigma(t, x) * sigma(t, x) * d2f_dx2(t, x);
        };
        
        // Diffusion coefficient: σ∂f/∂x
        result.diffusion_coeff = [df_dx, sigma](T t, T x) -> T {
            return sigma(t, x) * df_dx(t, x);
        };
        
        return result;
    }
    
    // Specific case: f(X_t) = X_t^n (power function)
    static ItoResult power_function_ito(T n,
                                        std::function<T(T, T)> mu,
                                        std::function<T(T, T)> sigma) {
        
        return apply_ito_lemma(
                               [n](T t, T x) -> T { return std::pow(x, n); },                    // f(t,x) = x^n
                               [](T t, T x) -> T { return T(0); },                              // ∂f/∂t = 0
                               [n](T t, T x) -> T { return n * std::pow(x, n-1); },             // ∂f/∂x = nx^(n-1)
                               [n](T t, T x) -> T { return n * (n-1) * std::pow(x, n-2); },     // ∂²f/∂x² = n(n-1)x^(n-2)
                               mu, sigma
                               );
    }
    
    // Specific case: f(X_t) = log(X_t)
    static ItoResult log_function_ito(std::function<T(T, T)> mu,
                                      std::function<T(T, T)> sigma) {
        
        return apply_ito_lemma(
                               [](T t, T x) -> T { return std::log(x); },                       // f(t,x) = log(x)
                               [](T t, T x) -> T { return T(0); },                              // ∂f/∂t = 0
                               [](T t, T x) -> T { return T(1) / x; },                          // ∂f/∂x = 1/x
                               [](T t, T x) -> T { return T(-1) / (x * x); },                   // ∂²f/∂x² = -1/x²
                               mu, sigma
                               );
    }
    
    // Specific case: f(X_t) = exp(X_t)
    static ItoResult exp_function_ito(std::function<T(T, T)> mu,
                                      std::function<T(T, T)> sigma) {
        
        return apply_ito_lemma(
                               [](T t, T x) -> T { return std::exp(x); },                       // f(t,x) = exp(x)
                               [](T t, T x) -> T { return T(0); },                              // ∂f/∂t = 0
                               [](T t, T x) -> T { return std::exp(x); },                       // ∂f/∂x = exp(x)
                               [](T t, T x) -> T { return std::exp(x); },                       // ∂²f/∂x² = exp(x)
                               mu, sigma
                               );
    }
    
    // Geometric Brownian Motion exact solution using Ito's lemma
    // For dX_t = μX_t dt + σX_t dW_t, we get X_t = X_0 exp((μ - σ²/2)t + σW_t)
    static std::vector<T> geometric_brownian_exact_solution(
                                                            T x0, T mu, T sigma, const std::vector<T>& brownian_path,
                                                            const std::vector<T>& time_grid) {
        
        if (brownian_path.size() != time_grid.size()) {
            throw std::invalid_argument("Brownian path and time grid must have same size");
        }
        
        std::vector<T> solution;
        solution.reserve(brownian_path.size());
        
        T t0 = time_grid.empty() ? T(0) : time_grid[0];
        
        for (size_t i = 0; i < brownian_path.size(); ++i) {
            T t = time_grid[i] - t0;
            T W_t = brownian_path[i];
            
            T exponent = (mu - sigma * sigma / 2) * t + sigma * W_t;
            solution.push_back(x0 * std::exp(exponent));
        }
        
        return solution;
    }
    
    // Verify Ito's lemma numerically
    static T verify_ito_lemma_numerically(
                                          std::function<T(T, T)> f,
                                          const ItoResult& ito_result,
                                          const std::vector<T>& process_path,
                                          const std::vector<T>& time_grid) {
        
        if (process_path.size() != time_grid.size() || process_path.size() < 2) {
            return T(0);
        }
        
        T direct_change = f(time_grid.back(), process_path.back()) -
        f(time_grid.front(), process_path.front());
        
        T ito_prediction = T(0);
        for (size_t i = 0; i < process_path.size() - 1; ++i) {
            T t = time_grid[i];
            T x = process_path[i];
            T dt = time_grid[i + 1] - time_grid[i];
            T dx = process_path[i + 1] - process_path[i];
            
            ito_prediction += ito_result.drift_coeff(t, x) * dt +
            ito_result.diffusion_coeff(t, x) * dx;
        }
        
        return std::abs(direct_change - ito_prediction); // Error measure
    }
};

} // namespace stochastic
