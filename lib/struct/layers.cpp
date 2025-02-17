#include "layers.h"
#include "helpers.h"

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
    step = computeLearningRate(tp, rho, alpha, k);
    
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

// ------------------------------ CONVOLUTION LAYER ---------------------------

/**
 * @brief Randomly initializes the convolution kernel.
 *
 * If q is 0, a square kernel of size p x p is created.
 * Each element is set to a random value between -1 and 1.
 *
 * @param p Number of rows.
 * @param q Number of columns (if 0, set equal to p).
 */
 void Convolution::randomK(Integer p, Integer q) {
    if (q == 0)
        q = p;
    K.resize(p);
    for (auto &row : K) {
        row.resize(q);
        for (auto &val : row) {
            val = static_cast<Reel>(rand()) / RAND_MAX * 2.0f - 1.0f;
        }
    }
}

/**
 * @brief Performs forward propagation for the convolution layer.
 *
 * Computes the convolution of the previous layer's input using kernel K.
 * Supports same-size convolution (with zero-padding) if same_size is true,
 * and valid convolution (without padding) otherwise.
 */
void Convolution::prop() {
    Layer* prev = this->prevL();
    assert(prev != nullptr);

    Integer in_rows = prev->dims[0];
    Integer in_cols = prev->dims[1];
    const RVector &input = prev->X;
    Integer k_rows = K.size();
    Integer k_cols = (k_rows > 0 ? K[0].size() : 0);
    RVector output;

    if (same_size) {
        dims[0] = in_rows;
        dims[1] = in_cols;
        output.resize(in_rows * in_cols);
        for (Integer i = 0; i < in_rows; i++) {
            for (Integer j = 0; j < in_cols; j++) {
                output[i * in_cols + j] = kerPad(input, in_rows, in_cols, i, j, K);
            }
        }
    } else {
        Integer start_i = (i0 > 0 ? i0 - 1 : 0);
        Integer start_j = (j0 > 0 ? j0 - 1 : 0);
        Integer out_rows = (in_rows - start_i >= k_rows) ? ((in_rows - start_i - k_rows) / mu + 1) : 0;
        Integer out_cols = (in_cols - start_j >= k_cols) ? ((in_cols - start_j - k_cols) / nu + 1) : 0;
        dims[0] = out_rows;
        dims[1] = out_cols;
        output.resize(out_rows * out_cols);
        for (Integer i = 0; i < out_rows; i++) {
            for (Integer j = 0; j < out_cols; j++) {
                output[i * out_cols + j] = kerNoPad(input, in_rows, in_cols, i, j, K, start_i, start_j, mu, nu);
            }
        }
    }
    this->X = output;
}

/**
 * @brief Performs backward propagation for the convolution layer.
 *
 */
void Convolution::backprop() {
    Layer* prev = this->prevL();
    assert(prev != nullptr);

    Integer in_rows = prev->dims[0];
    Integer in_cols = prev->dims[1];
    const RVector &input = prev->X;
    Integer out_rows = dims[0];
    Integer out_cols = dims[1];
    Integer k_rows = K.size();
    Integer k_cols = (k_rows > 0 ? K[0].size() : 0);

    // Initialize GradP (kernel gradient matrix)
    if (GradP.size() != k_rows * k_cols) {
        GradP = RMatrix(k_rows, k_cols, 0);
    } else {
        for (auto &val : GradP)
            val = 0;
    }
    RVector gradInput(in_rows * in_cols, 0);
    const RVector &gradOutput = this->GradX;

    if (same_size) {
        for (Integer i = 0; i < out_rows; i++) {
            for (Integer j = 0; j < out_cols; j++) {
                Reel grad_val = gradOutput[i * out_cols + j];
                accumulateGradPad(input, in_rows, in_cols, i, j, K, GradP, gradInput, grad_val);
            }
        }
    } else {
        Integer start_i = (i0 > 0 ? i0 - 1 : 0);
        Integer start_j = (j0 > 0 ? j0 - 1 : 0);
        for (Integer i = 0; i < out_rows; i++) {
            for (Integer j = 0; j < out_cols; j++) {
                Reel grad_val = gradOutput[i * out_cols + j];
                accumulateGradNoPad(input, in_rows, in_cols, i, j, K, GradP, gradInput, grad_val, start_i, start_j, mu, nu);
            }
        }
    }

    // Propagate the input gradient to the previous layer.
    if (prev->GradX.size() != gradInput.size())
        prev->GradX = gradInput;
    else {
        for (size_t idx = 0; idx < gradInput.size(); idx++)
            prev->GradX[idx] += gradInput[idx];
    }
}

/**
 * @brief Updates the kernel parameters using gradient descent.
 *
 * The learning rate is computed based on the step type (tp) and parameters rho, alpha, and iteration count k.
 *
 * @param tp Step type.
 * @param rho Decay rate.
 * @param alpha Base learning rate.
 * @param k Current iteration.
 */
void Convolution::majparameters(int tp, Reel rho, Reel alpha, Integer k) {
    Reel lr = computeLearningRate(tp, rho, alpha, k);
    Integer k_rows = K.size();
    if (k_rows == 0)
        return;
    Integer k_cols = K[0].size();
    for (Integer i = 0; i < k_rows; i++) {
        for (Integer j = 0; j < k_cols; j++) {
            K[i][j] -= lr * GradP(i, j);
        }
    }
}

/**
 * @brief Prints the details of the convolution layer.
 *
 * Outputs kernel dimensions, strides, starting indices, and the kernel values.
 */
void Convolution::print(std::ostream& out) const {
    out << "Convolution Layer:" << "\n";
    out << "Kernel dimensions: " << K.size() << " x " 
        << (K.empty() ? 0 : K[0].size()) << "\n";
    out << "Strides: mu = " << mu << ", nu = " << nu << "\n";
    out << "Starting indices: i0 = " << i0 << ", j0 = " << j0 << "\n";
    out << "Preserve size (same_size): " 
        << (same_size ? "true" : "false") << "\n";
    out << "Kernel values:" << "\n";
    for (const auto &row : K) {
        for (const auto &val : row) {
            out << val << " ";
        }
        out << "\n";
    }
}