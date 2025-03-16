#include "helpers.hpp"


namespace utils { 

// ---------------------------- ACTIVATION FUNCTIONS ----------------------------
Reel relu(Reel x) {
    return (x > 0) ? x : 0;
}

Reel d_relu(Reel x) {
    return (x > 0) ? 1 : 0;
}

Reel hyper_tan(Reel x) {
    return std::tanh(x);
}

Reel d_hyper_tan(Reel x) {
    return 1 - std::tanh(x) * std::tanh(x);
}

Reel abs_hyper_tan(Reel x) {
    return std::abs(std::tanh(x));
}

Reel d_abs_hyper_tan(Reel x) {
    return (1 - std::tanh(x) * std::tanh(x)) * (x > 0 ? 1 : -1);
}

Reel sigmoid(Reel x) {
    return 1 / (1 + std::exp(-x));
}

Reel d_sigmoid(Reel x) {
    Reel s = sigmoid(x);
    return s * (1 - s);
}


Reel computeLearningRate(TypeStep tp, Reel rho, Reel alpha, Integer k) {
    Reel step = 0;
    switch (tp) {
        case _fixed:
            step = rho;
            break;
        case _linear:
            step = rho / (1 + alpha * k);
            break;
        case _quadratic:
            step = rho / (1 + alpha * k * k);
            break;
        case _exponential:
            step = rho * std::exp(-alpha * k);
            break;
        default:
            throw std::invalid_argument("Invalid step type");
    }
    return step;
}

// Auxiliary function to shuffle two vectors in unison
void shuffle_data(std::vector<RVector>& Es, std::vector<RVector>& Ss) {
    // Create a vector of indices
    std::vector<size_t> indices(Es.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    // Shuffle the indices
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Reorder Es and Ss using the shuffled indices
    std::vector<RVector> shuffled_Es(Es.size());
    std::vector<RVector> shuffled_Ss(Ss.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        shuffled_Es[i] = Es[indices[i]];
        shuffled_Ss[i] = Ss[indices[i]];
    }

    // Replace the original vectors with the shuffled ones
    Es = std::move(shuffled_Es);
    Ss = std::move(shuffled_Ss);
}

// Function to compute the outer product of two vectors
// res = a * b^T
// size(a) = m, size(b) = n, size(res) = m x n
RMatrix outerProduct(const RVector& a, const RVector& b) {
    int m = a.size();
    int n = b.size();

    RMatrix result(m, n);

    // Compute the outer product
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result(i, j) = a[i] * b[j];
        }
    }
    return result;
}

RMatrix conv(const RMatrix& X, const RMatrix& ker){

    int n = X.rows;
    if (n != X.cols){
        throw std::invalid_argument("Matrix X must be square");
    }

    int m = ker.rows;
    if (m != ker.cols){
        throw std::invalid_argument("Matrix ker must be square");
    }
    RMatrix res(n-m+1, n-m+1);
    for (int i = 0; i < n-m+1; i++){
        for (int j = 0; j < n-m+1; j++){
            res(i, j) = 0;
            for (int k = 0; k < m; k++){
                for (int l = 0; l < m; l++){
                    res(i, j) += X(i+k, j+l) * ker(k, l);
                }
            }
        }
    }
    return res;
}

// ---------------------------- LOSS FUNCTIONS ----------------------------
Reel moindre_carre(const RVector& y, const RVector& y_pred) {
    Reel res = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        res += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);
    }
    return res / 2;
}

RVector d_moindre_carre(const RVector& y, const RVector& y_pred) {
    RVector res(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        res[i] = -(y[i] - y_pred[i]);
    }
    return res;
}

Reel moindre_abs(const RVector& y, const RVector& y_pred) {
    Reel res = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        res += std::abs(y[i] - y_pred[i]);
    }
    return res;
}

RVector d_moindre_abs(const RVector& y, const RVector& y_pred) {
    RVector res(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        res[i] = (y_pred[i] > y[i]) ? 1 : (y_pred[i] < y[i]) ? -1 : 0;
    }
    return res;
}

Reel entropie_croisee(const RVector& y, const RVector& y_pred) {
    Reel res = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        res += y[i] * std::log(y_pred[i]);
    }
    return -res;
}

RVector d_entropie_croisee(const RVector& y, const RVector& y_pred) {
    RVector res(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        res[i] = -y[i] / y_pred[i];
    }
    return res;
}


} // namespace utils


std::ostream& operator<<(std::ostream& os, TypeLayer type) {
    switch (type) {
        case _nondefini:    return os << "Undefined";
        case _Entry:        return os << "Entry Layer";
        case _Convolution:  return os << "Convolution Layer";
        case _Reduction:    return os << "Reduction Layer";
        case _Activation:   return os << "Activation Layer";
        case _Dense:        return os << "Dense Layer";
        case _Loss:         return os << "Loss Layer";
        default:            return os << "Unknown Layer";
    }
}