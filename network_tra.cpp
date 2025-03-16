#include "layers.hpp"
#include <algorithm> 


void Network::forwardprop(const RVector& x_i, const RVector& y_i){
    
    layers[0]->X = x_i; // Set the input layer to the input vector x_i
    Loss *last = layers[layers.size() - 1]; // Get the last layer
    last.set_vref(y_i); // Set the reference vector of the loss layer to the target vector y_i
    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i]->forwardprop(); // Update the X values of the layers
    }
}

void Network::backprop() {
    for (size_t i = layers.size() - 1; i >= 0; --i) {
        if (layers[i]->params == true) {
            layers[i]->backprop();
        }
    }
}

void Network::majParametres(TypePas tp, Reel rho, Reel alpha, Entier k) {
    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i]->majparameters(tp, rho, alpha, k);
    }
}


// Auxiliary function to shuffle two vectors in unison
void shuffle_data(vector<RVector>& Es, vector<RVector>& Ss) {
    // Create a vector of indices
    vector<size_t> indices(Es.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    // Shuffle the indices
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Reorder Es and Ss using the shuffled indices
    vector<RVector> shuffled_Es(Es.size());
    vector<RVector> shuffled_Ss(Ss.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        shuffled_Es[i] = Es[indices[i]];
        shuffled_Ss[i] = Ss[indices[i]];
    }

    // Replace the original vectors with the shuffled ones
    Es = shuffled_Es;
    Ss = shuffled_Ss;
}

vector<vector<RVector>> Network::batch_split(const vector<RVector>& Es, const vector<RVector>& Ss) {
    
    vector<RVector> shuffled_Es = Es;
    vector<RVector> shuffled_Ss = Ss;

    shuffle_data(shuffled_Es, shuffled_Ss);

    vector<vector<RVector>> batches;
    size_t n = shuffled_Es.size();
    size_t b = batch_size;
    size_t m = n / b;

    for (size_t i = 0; i < m; ++i) {
        vector<RVector> E(b);
        vector<RVector> S(b);

        for (size_t j = 0; j < b; ++j) {
            E[j] = shuffled_Es[i * b + j];
            S[j] = shuffled_Ss[i * b + j];
        }
        batches.push_back({E, S});
    }

    return batches;
}

void Network::train_batch(const vector<RVector>& Es, const vector<RVector>& Ss) {
    for (size_t i = 0; i < Es.size(); ++i) {
        RVector& x_i = Es[i];
        RVector& y_i = Ss[i];
        forwardprop(x_i, y_i);
        backprop(); // update the gradient
    }
}


void Network::train(const std::vector<RVector>& Es, const std::vector<RVector>& Ss, TypePas tp, Reel rho, Reel alpha) {
    vector<vector<RVector>> batches = batch_split(Es, Ss);
    for (size_t i = 0; i < batches.size(); ++i) {
        train_batch(batches[i][0], batches[i][1], tp, rho, alpha);
        majParametres(tp, rho, alpha, i); // update weights after each batch
    }
}