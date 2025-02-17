#include "layers.h"

//------------------------------ GENERIC LAYER --------------------------------

Layer* Layer::nextL() {
    // if the layer is the last one or the network is not set, return nullptr
    if (network == nullptr || index + 1 >= network->layers.size()) {
        return nullptr;
    }
    return network->layers[index + 1];
}

Layer* Layer::prevL() {
    // if the layer is the first one or the network is not set, return nullptr
    if (network == nullptr || index == 0) {
        return nullptr;
    }
    return network->layers[index - 1];
}

void Layer::print(std::ostream& out) const {
    out << "Layer type: " << type << "\n";
    out << "Dimensions: (" << dims[0] << ", " << dims[1] << ", " << dims[2] << ")\n";
    out << "Index: " << index << "\n";
    out << "Number of parameters: " << (parameters ? GradP.size() : 0) << "\n";
}


// ------------------------------ DENSE LAYER ---------------------------------

/**
 * @brief Clones the Dense layer.
 * This function creates a deep copy of the current Dense layer.
 */
 Dense* Dense::clone() const {
    Dense* newDense = new Dense(*this); 
    newDense->C = this->C; 
    return newDense;
}

/**
 * @brief Forward propagation for the Dense layer.
 *  Computes: X = C * (previous layer's X)
 */
void Dense::forwardprop() {
    Layer* prev = this->prevL(); // get pointer to the previous layer
    if (!prev) {
        std::cerr << "Dense::forwardprop: no previous layer found." << std::endl;
        return;
    }
    X = C * prev->X;
}

/**
 * @brief Performs backward propagation for the Dense layer.
 * 
 * 1. Propagates the error backward: previous->GradX += C^T * GradX.
 * 2. Computes the gradient of the weights: GradC = outer_product(GradX, previous->X)
*/
void Dense::backprop() {
    Layer* prev = this->prevL(); // Get pointer to the previous layer
    if (!prev) {
        std::cerr << "Dense::backprop: no previous layer found." << std::endl;
        return;
    }
    
    // Backpropagate the error: previous->GradX += C^T * GradX
    prev->GradX += transpose(C) * GradX;
    
    // Compute the gradient with respect to the weights: gradP_ij = prev_j * GradX_i
    gradP = outer_product(GradX, prev->X);
}


/**
 * @brief Updates the parameters (weights) of the Dense layer.
 * 
 * This function updates the weights of the Dense layer using the computed gradients (GradP).
 * The update rule depends on the type of step (tp), learning rate (rho), momentum (alpha),
 * and the current iteration (k).
 * 
 * @param tp The type of step (e.g., fixed, linear, quadratic, exponential).
 * @param rho The learning rate.
 * @param alpha The momentum factor.
 * @param k The current iteration number.
 */
 void Dense::majparameters(int tp, Reel rho, Reel alpha, Integer k) {
    Reel step;
    switch(tp) {
        case _fixed:
            step = alpha;
            break;
        case _linear:
            step = alpha / k;
            break;
        case _quadratic:
            step = alpha / (k * k);
            break;
        case _exponential:
            step = alpha * std::exp(-static_cast<Reel>(k));
            break;
        default:
            step = alpha;
            break;
    }
    
    size_t mC = dims[0];  // number of rows (outputs)
    size_t nC = dims[1];  // number of columns (inputs)
    
    // update each weight in the matrix C.
    for (size_t i = 0; i < mC; ++i) {
        for (size_t j = 0; j < nC; ++j) {
            GradPm(i, j) = rho * GradPm(i, j) - step * GradP(i, j);
            C(i, j) += GradPm(i, j);
        }
    }
}



/**
 * @brief Prints the details of the Dense layer.
 */
void Dense::print(std::ostream& out) const {
    out << "Dense Layer: " << "\n";
    out << "Weights: \n";
    for (const auto& row : C) {
        for (const auto& val : row) {
            out << val << " ";
        }
        out << "\n";
    }
}