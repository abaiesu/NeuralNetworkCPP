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
void shuffle_data(std::vector<RTensor>& Es, std::vector<RTensor>& Ss) {
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
    std::vector<RTensor> shuffled_Es(Es.size());
    std::vector<RTensor> shuffled_Ss(Ss.size());
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
RTensor outerProduct(const RTensor& a, const RTensor& b) {

    // check that both are vectors
    if (a.rank() != 1 || b.rank() != 1){
        throw std::invalid_argument("(outerProd) Both tensors must be vectors");
    }

    Integer m = a.size();
    Integer n = b.size();

    RTensor result(m, n);

    // Compute the outer product
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result(i, j) = a[i] * b[j];
        }
    }
    return result;
}

RTensor convolve(const RTensor& ker, const RTensor& X){

    // check that both have rank 2 = both are matrices
    if (ker.rank() != 2 || X.rank() != 2){
        throw std::invalid_argument("Both tensors must be matrices");
    }

    Integer n = X.dims(0);
    if (n != X.dims(1)){
        throw std::invalid_argument("Matrix X must be square");
    }

    Integer m = ker.dims(0);
    if (m != ker.dims(1)){
        throw std::invalid_argument("Matrix ker must be square");
    }

    if (m > n){
        throw std::invalid_argument("Matrix ker must be smaller than matrix X");
    }

    Integer new_n = n - m + 1;
    RTensor res(new_n, new_n); // implicitly rank 2
    for (Integer i = 0; i < new_n; i++){
        for (Integer j = 0; j < new_n; j++){
            res(i, j) = 0;
            for (Integer k = 0; k < m; k++){
                for (Integer l = 0; l < m; l++){
                    res(i, j) += X(i+k, j+l) * ker(k, l);
                }
            }
        }
    }
    return res;
}

// ---------------------------- LOSS FUNCTIONS ----------------------------
Reel moindre_carre(const RTensor& y, const RTensor& y_pred) {
    // make sure both are vectors
    if (y.rank() != 1 || y_pred.rank() != 1){
        throw std::invalid_argument("(moidre_carre) Both tensors must be vectors");
    }
    Reel res = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        res += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);
    }
    return res / 2;
}

RTensor d_moindre_carre(const RTensor& y, const RTensor& y_pred) {
    if (y.rank() != 1 || y_pred.rank() != 1){
        throw std::invalid_argument("(d_moindre_carre) Both tensors must be vectors");
    }
    RTensor res(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        res[i] = -(y[i] - y_pred[i]);
    }
    return res;
}

Reel moindre_abs(const RTensor& y, const RTensor& y_pred) {
    if (y.rank() != 1 || y_pred.rank() != 1){
        throw std::invalid_argument("(moidre_abs) Both tensors must be vectors");
    }
    Reel res = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        res += std::abs(y[i] - y_pred[i]);
    }
    return res;
}

RTensor d_moindre_abs(const RTensor& y, const RTensor& y_pred) {
    if (y.rank() != 1 || y_pred.rank() != 1){
        throw std::invalid_argument("(d_moindre_abs) Both tensors must be vectors");
    }
    RTensor res(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        res[i] = (y_pred[i] > y[i]) ? 1 : (y_pred[i] < y[i]) ? -1 : 0;
    }
    return res;
}

Reel entropie_croisee(const RTensor& y, const RTensor& y_pred) {
    if (y.rank() != 1 || y_pred.rank() != 1){
        throw std::invalid_argument("(entropie_croisee) Both tensors must be vectors");
    }
    Reel res = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        res += y[i] * std::log(y_pred[i]);
    }
    return -res;
}

RTensor d_entropie_croisee(const RTensor& y, const RTensor& y_pred) {
    if (y.rank() != 1 || y_pred.rank() != 1){
        throw std::invalid_argument("(d_entropie_croisee) Both tensors must be vectors");
    }
    RTensor res(y.size());
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
        case _Pool:         return os << "Pooling Layer";
        case _Activation:   return os << "Activation Layer";
        case _Dense:        return os << "Dense Layer";
        case _Loss:         return os << "Loss Layer";
        case _Flatten:      return os << "Flatten Layer";
        default:            return os << "Unknown Layer";
    }
}