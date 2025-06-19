//
//  statistics.hpp
//  StochasticForge
//
//  Created by Suad Demiri on 19.06.25.
//

// include/stochastic/core/statistics.hpp
#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace stochastic {

template<typename T>
class Statistics {
public:
    static T mean(const std::vector<T>& data) {
        if (data.empty()) return T{};
        return std::accumulate(data.begin(), data.end(), T{}) / data.size();
    }
    
    static T variance(const std::vector<T>& data, bool sample = true) {
        if (data.size() <= 1) return T{};
        
        const T mu = mean(data);
        T sum_sq = 0;
        for (const auto& x : data) {
            const T diff = x - mu;
            sum_sq += diff * diff;
        }
        
        const size_t denominator = sample ? data.size() - 1 : data.size();
        return sum_sq / denominator;
    }
    
    static T standard_deviation(const std::vector<T>& data, bool sample = true) {
        return std::sqrt(variance(data, sample));
    }
    
    static T skewness(const std::vector<T>& data) {
        if (data.size() < 3) return T{};
        
        const T mu = mean(data);
        const T sigma = standard_deviation(data, false);
        if (sigma == 0) return T{};
        
        T sum_cubed = 0;
        for (const auto& x : data) {
            const T z = (x - mu) / sigma;
            sum_cubed += z * z * z;
        }
        
        return sum_cubed / data.size();
    }
    
    static T kurtosis(const std::vector<T>& data) {
        if (data.size() < 4) return T{};
        
        const T mu = mean(data);
        const T sigma = standard_deviation(data, false);
        if (sigma == 0) return T{};
        
        T sum_fourth = 0;
        for (const auto& x : data) {
            const T z = (x - mu) / sigma;
            const T z_sq = z * z;
            sum_fourth += z_sq * z_sq;
        }
        
        return sum_fourth / data.size() - 3; // Excess kurtosis
    }
    
    static T covariance(const std::vector<T>& x, const std::vector<T>& y, bool sample = true) {
        if (x.size() != y.size() || x.size() <= 1) return T{};
        
        const T mean_x = mean(x);
        const T mean_y = mean(y);
        
        T sum_prod = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            sum_prod += (x[i] - mean_x) * (y[i] - mean_y);
        }
        
        const size_t denominator = sample ? x.size() - 1 : x.size();
        return sum_prod / denominator;
    }
    
    static T correlation(const std::vector<T>& x, const std::vector<T>& y) {
        const T cov = covariance(x, y, false);
        const T std_x = standard_deviation(x, false);
        const T std_y = standard_deviation(y, false);
        
        if (std_x == 0 || std_y == 0) return T{};
        return cov / (std_x * std_y);
    }
    
    static T moment(const std::vector<T>& data, int order) {
        if (data.empty() || order < 0) return T{};
        
        T sum = 0;
        for (const auto& x : data) {
            sum += std::pow(x, order);
        }
        return sum / data.size();
    }
    
    static T central_moment(const std::vector<T>& data, int order) {
        if (data.empty() || order < 0) return T{};
        if (order == 0) return T{1};
        if (order == 1) return T{0};
        
        const T mu = mean(data);
        T sum = 0;
        for (const auto& x : data) {
            sum += std::pow(x - mu, order);
        }
        return sum / data.size();
    }
    
    // Quantiles and percentiles
    static T quantile(std::vector<T> data, T p) {
        if (data.empty() || p < 0 || p > 1) return T{};
        
        std::sort(data.begin(), data.end());
        
        if (p == 0) return data.front();
        if (p == 1) return data.back();
        
        const T index = p * (data.size() - 1);
        const size_t lower = static_cast<size_t>(std::floor(index));
        const size_t upper = static_cast<size_t>(std::ceil(index));
        
        if (lower == upper) {
            return data[lower];
        }
        
        const T weight = index - lower;
        return data[lower] * (1 - weight) + data[upper] * weight;
    }
    
    static T median(std::vector<T> data) {
        return quantile(std::move(data), 0.5);
    }
};

} // namespace stochastic
