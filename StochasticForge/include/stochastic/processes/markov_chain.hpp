//
//  markov_chain.hpp
//  StochasticForge
//
//  Created by Suad Demiri on 19.06.25.
//

// include/stochastic/processes/markov_chain.hpp
#pragma once
#include <vector>
#include <unordered_map>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace stochastic {

template<typename State>
class MarkovChain {
private:
    std::vector<State> states_;
    std::vector<std::vector<double>> transition_matrix_;
    std::unordered_map<State, size_t> state_to_index_;
    State current_state_;
    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<double> uniform_dist_;
    
public:
    MarkovChain(const std::vector<State>& states,
                const std::vector<std::vector<double>>& transition_matrix,
                const State& initial_state,
                unsigned seed = std::random_device{}())
    : states_(states), transition_matrix_(transition_matrix),
    current_state_(initial_state), rng_(seed), uniform_dist_(0.0, 1.0) {
        
        // Validate inputs
        if (states_.empty()) {
            throw std::invalid_argument("States vector cannot be empty");
        }
        
        if (transition_matrix_.size() != states_.size()) {
            throw std::invalid_argument("Transition matrix size must match number of states");
        }
        
        // Build state-to-index mapping
        for (size_t i = 0; i < states_.size(); ++i) {
            state_to_index_[states_[i]] = i;
        }
        
        // Validate initial state
        if (state_to_index_.find(initial_state) == state_to_index_.end()) {
            throw std::invalid_argument("Initial state not found in states vector");
        }
        
        // Validate transition matrix
        validate_transition_matrix();
    }
    
    State get_current_state() const { return current_state_; }
    
    void set_current_state(const State& state) {
        if (state_to_index_.find(state) == state_to_index_.end()) {
            throw std::invalid_argument("State not found in chain");
        }
        current_state_ = state;
    }
    
    State next_state() {
        size_t current_index = state_to_index_[current_state_];
        const auto& probabilities = transition_matrix_[current_index];
        
        double random_val = uniform_dist_(rng_);
        double cumulative_prob = 0.0;
        
        for (size_t i = 0; i < probabilities.size(); ++i) {
            cumulative_prob += probabilities[i];
            if (random_val <= cumulative_prob) {
                current_state_ = states_[i];
                return current_state_;
            }
        }
        
        // Fallback (should not reach here if probabilities sum to 1)
        current_state_ = states_.back();
        return current_state_;
    }
    
    std::vector<State> simulate(size_t steps) {
        std::vector<State> path;
        path.reserve(steps + 1);
        path.push_back(current_state_);
        
        for (size_t i = 0; i < steps; ++i) {
            path.push_back(next_state());
        }
        
        return path;
    }
    
    std::vector<State> simulate_from(const State& start_state, size_t steps) {
        State original_state = current_state_;
        set_current_state(start_state);
        auto path = simulate(steps);
        current_state_ = original_state;  // Restore original state
        return path;
    }
    
    // Calculate stationary distribution using power method
    std::vector<double> stationary_distribution(double tolerance = 1e-10, size_t max_iterations = 1000) const {
        size_t n = states_.size();
        std::vector<double> pi(n, 1.0 / n);  // Start with uniform distribution
        
        for (size_t iter = 0; iter < max_iterations; ++iter) {
            std::vector<double> pi_new(n, 0.0);
            
            // π_new = π * P
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    pi_new[j] += pi[i] * transition_matrix_[i][j];
                }
            }
            
            // Check convergence
            double max_diff = 0.0;
            for (size_t i = 0; i < n; ++i) {
                max_diff = std::max(max_diff, std::abs(pi_new[i] - pi[i]));
            }
            
            pi = pi_new;
            
            if (max_diff < tolerance) {
                break;
            }
        }
        
        return pi;
    }
    
    // Get n-step transition probabilities
    std::vector<std::vector<double>> n_step_transition_matrix(size_t n) const {
        if (n == 0) {
            // Return identity matrix
            size_t size = states_.size();
            std::vector<std::vector<double>> identity(size, std::vector<double>(size, 0.0));
            for (size_t i = 0; i < size; ++i) {
                identity[i][i] = 1.0;
            }
            return identity;
        }
        
        if (n == 1) {
            return transition_matrix_;
        }
        
        // Matrix exponentiation by repeated squaring
        auto result = transition_matrix_;
        auto base = transition_matrix_;
        n--;
        
        while (n > 0) {
            if (n % 2 == 1) {
                result = matrix_multiply(result, base);
            }
            base = matrix_multiply(base, base);
            n /= 2;
        }
        
        return result;
    }
    
    const std::vector<State>& get_states() const { return states_; }
    const std::vector<std::vector<double>>& get_transition_matrix() const { return transition_matrix_; }
    
private:
    void validate_transition_matrix() {
        for (size_t i = 0; i < transition_matrix_.size(); ++i) {
            if (transition_matrix_[i].size() != states_.size()) {
                throw std::invalid_argument("Each row of transition matrix must have same size as states vector");
            }
            
            double row_sum = std::accumulate(transition_matrix_[i].begin(), transition_matrix_[i].end(), 0.0);
            if (std::abs(row_sum - 1.0) > 1e-10) {
                throw std::invalid_argument("Each row of transition matrix must sum to 1");
            }
            
            for (double prob : transition_matrix_[i]) {
                if (prob < 0.0 || prob > 1.0) {
                    throw std::invalid_argument("All transition probabilities must be in [0, 1]");
                }
            }
        }
    }
    
    std::vector<std::vector<double>> matrix_multiply(
                                                     const std::vector<std::vector<double>>& A,
                                                     const std::vector<std::vector<double>>& B) const {
                                                         
                                                         size_t n = A.size();
                                                         std::vector<std::vector<double>> C(n, std::vector<double>(n, 0.0));
                                                         
                                                         for (size_t i = 0; i < n; ++i) {
                                                             for (size_t j = 0; j < n; ++j) {
                                                                 for (size_t k = 0; k < n; ++k) {
                                                                     C[i][j] += A[i][k] * B[k][j];
                                                                 }
                                                             }
                                                         }
                                                         
                                                         return C;
                                                     }
};

} // namespace stochastic
