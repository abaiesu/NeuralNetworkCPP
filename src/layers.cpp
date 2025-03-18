#include "layers.hpp"
#include "helpers.hpp"
#include <random> 

//------------------------------ GENERIC LAYER --------------------------------

Layer* Layer::nextL() {
    if (network == nullptr || index >= network->getLayers().size() - 1) {
        std::cerr << "Layer::nextL: no next layer found." << std::endl;
        return nullptr;
    }
    return network->getLayers()[index + 1];
}

Layer* Layer::prevL() {
    if (network == nullptr || index == 0) {
        std::cerr << "Layer::prevL: no previous layer found." << std::endl;
        return nullptr;
    }
    return network->getLayers()[index - 1];
}

void Layer::print(std::ostream& out) const {
    out << "Layer type: " << type << ", Index: " << index;
    out << ", Dimensions: (" << dims[0] << ", " << dims[1] << ", " << dims[2] << ")";
    out << ", Size of X: " << X.size() << "\n";
}


/**
 * @brief Updates the weights of the layer for 1 iteration of gradient descent
 * 
 * @param tp The type of step (e.g., fixed, linear, quadratic, exponential).
 * @param rho The learning rate.
 * @param alpha The momentum factor.
 * @param k The current iteration number.
 */
 void Layer::majparameters(TypeStep tp, Reel rho, Reel alpha, Integer k) {
    
    if (params){
        Reel step = rho; //utils::computeLearningRate(tp, rho, alpha, k);
        Reel v = 0;
        GradWm *= step; // update the gradient
        W -= GradWm; // update the weights
        if(GradWm.rank() == 2){
            GradWm = RTensor(GradWm.dims(0), GradWm.dims(1)); // reset the gradient
        }
        else if(GradWm.rank() == 4){
            GradWm = RTensor(GradWm.dims(0), GradWm.dims(1), GradWm.dims(2), GradWm.dims(3)); // reset the gradient
        }
        else{
            std::cerr << "Layer::majparameters: GradW rank is not 2 or 4, it is " << GradWm.rank() << std::endl;
        }
    }
}

// ------------------------------ DENSE LAYER ---------------------------------


void Dense::set_dims(Integer n_prev) {

    Integer n = dims[0];
    W = RTensor(n, n_prev); // from a space of size n_prev to a space of size n


    // Compute Glorot/Xavier uniform limit
    double limit = std::sqrt(6.0 / (n + n_prev));

    // Random initialization in [-limit, limit]
    for (Integer i = 0; i < n; ++i) {
        for (Integer j = 0; j < n_prev; ++j) {
            double r = (double)std::rand() / (double)RAND_MAX; // in [0,1]
            W(i, j) = r * 2.0 * limit - limit; // in [-limit, limit]
        }
    }

    GradW = RTensor(n, n_prev); // Empty matrix for gradient
    GradWm = RTensor(n, n_prev); // Empty matrix for average gradient
}


/**
 * @brief Clones the Dense layer.
 * This function creates a deep copy of the current Dense layer.
 */
 Dense* Dense::clone() const {
    Dense* newDense = new Dense(*this); 
    newDense->W = this->W; 
    return newDense;
}

/**
 * @brief Forward propagation for the Dense layer.
 *  Computes: X = C * (previous layer's X)
 */
void Dense::forwardprop() {
    Layer* prev = this->prevL(); // get pointer to the previous layer
    X = W * prev->X;
}

/**
 * @brief Performs backward propagation for the Dense layer.
 * 
 * 1. Propagates the error backward: previous->GradX += C^T * GradX.
 * 2. Computes the gradient of the weights: GradC = outer_product(GradX, previous->X)
*/
void Dense::backprop() {

    // Compute gradX 

    // if before the last layer (recall that the last layer is the loss layer)
    if (index == network->getLayers().size() - 2) {
        Loss *loss = dynamic_cast<Loss*>(network->getLayers().back()); // the loss is the last layer
        RTensor true_labal = loss->get_vref();
        RTensor pred_labal = this->X;
        GradX = loss->get_dfun_Loss()(true_labal, pred_labal);
    }else{
        Layer* next = this->nextL(); // get pointer to the next layer
        if (next->type == _Dense){
            GradX = dynamic_cast<Dense*>(next)->get_dX_dPrevX_Dense() * next->GradX; // only update the GradX
        }
        else if (next->type == _Activation){
            GradX = dynamic_cast<Activation*>(next)->get_dX_dPrevX_Dense() * next->GradX; // only update the GradX
        } else{
            std::cerr << "Dense::backprop: next layer is not Dense or Activation" << std::endl;
        }
    }
    // Compute gradW and gradWm
    Layer* prev = this->prevL(); // get pointer to the previous layer
    GradW = utils::outerProduct(GradX, prev->X);
    GradWm += GradW;
}


// ------------------------------ LOSS LAYER ---------------------------

// Set function pointers based on TypeLoss
void Loss::setFunPtr() {
    switch (typeP) {
        case _moindre_carre:
            fun_Loss = utils::moindre_carre;
            dfun_Loss = utils::d_moindre_carre;
            break;
        case _moindre_abs:
            fun_Loss = utils::moindre_abs;
            dfun_Loss = utils::d_moindre_abs;
            break;
        case _entropie_croisee:
            fun_Loss = utils::entropie_croisee;
            dfun_Loss = utils::d_entropie_croisee;
            break;
        case _cat_cross_entropy:
            fun_Loss = utils::cat_cross_entropy;
            dfun_Loss = utils::d_cat_cross_entropy;
            break;
        default:
            throw std::invalid_argument("Invalid loss type");
    }
}


void Loss::forwardprop() {
    //std:: cout << "in loss forwardprop" << std::endl;
    Layer* prev = this->prevL(); // Get previous layer
    RTensor true_labal = vref;
    RTensor pred_labal = prev->X;
    this->X = RTensor(1);
    this->X(0) = fun_Loss(true_labal, pred_labal);
}
// ------------------------------- HELPERS -----------------------------------


// Helper function: backpropagation for a convolution next layer
void updateGradXFromConvolution(Layer* next, RTensor &GradX, const RTensor &X) {
    Integer next_channels = next->dims[2];
    Integer ker_size_next = dynamic_cast<Convolution*>(next)->ker_size;
    RTensor W_next = dynamic_cast<Convolution*>(next)->W;
    
    // Initialize GradX to zeros
    for (Integer i = 0; i < GradX.dims(0); i++) {
        for (Integer j = 0; j < GradX.dims(1); j++) {
            for (Integer k = 0; k < GradX.dims(2); k++) {
                GradX(i, j, k) = 0;
            }
        }
    }
    
    // Calculate the valid range for the convolution
    Integer n_prev = GradX.dims(0);
    Integer new_n = next->dims[0]; // Output size after convolution
    
    for (Integer i = 0; i < n_prev; i++) {
        for (Integer j = 0; j < n_prev; j++) {
            for (Integer k = 0; k < GradX.dims(2); k++) {
                // For each position in GradX where the kernel can be applied
                for (Integer c = 0; c < next_channels; c++) {
                    for (Integer u = 0; u < ker_size_next; u++) {
                        for (Integer v = 0; v < ker_size_next; v++) {
                            // Calculate the position in next->GradX
                            Integer next_i = i - u;
                            Integer next_j = j - v;
                            
                            // Only accumulate if the position is valid in next->GradX
                            if (next_i >= 0 && next_i < new_n && next_j >= 0 && next_j < new_n) {
                                // Use the flipped kernel for backpropagation
                                GradX(i, j, k) += W_next(ker_size_next - 1 - u, ker_size_next - 1 - v, k, c)
                                    * next->GradX(next_i, next_j, c);
                            }
                        }
                    }
                }
            }
        }
    }
}


// Helper function: backpropagation for a pooling next layer (mean and max)
void updateGradXFromPooling(Layer* next, RTensor &GradX, const RTensor &X) {
    Pool* poolLayer = dynamic_cast<Pool*>(next);
    Integer pool_size = poolLayer->p;
    Integer stride = poolLayer->stride;
    
    if (poolLayer->typeP == _meanPool) {
        for (Integer x = 0; x < GradX.dims(0); ++x) {
            for (Integer y = 0; y < GradX.dims(1); ++y) {
                for (Integer k = 0; k < GradX.dims(2); ++k) {
                    GradX(x, y, k) = 0; // Initialize gradient
                    for (Integer u = 0; u < pool_size; ++u) {
                        for (Integer v = 0; v < pool_size; ++v) {
                            Integer i = x * stride + u;
                            Integer j = y * stride + v;
                            if (i < next->dims[0] && j < next->dims[1]) {
                                GradX(x, y, k) += next->GradX(i, j, k) / (pool_size * pool_size);
                            }
                        }
                    }
                }
            }
        }
    } 
    else if (poolLayer->typeP == _maxPool) {
        for (Integer x = 0; x < GradX.dims(0); ++x) {
            for (Integer y = 0; y < GradX.dims(1); ++y) {
                for (Integer k = 0; k < GradX.dims(2); ++k) {
                    // Initialize the gradient at the pooling output
                    GradX(x, y, k) = 0;
                    Integer max_i = -1, max_j = -1;
                    float max_val = -std::numeric_limits<float>::infinity();
                    
                    for (Integer u = 0; u < pool_size; ++u) {
                        for (Integer v = 0; v < pool_size; ++v) {
                            Integer i = x * stride + u;
                            Integer j = y * stride + v;
                            if (i < next->dims[0] && j < next->dims[1]) {
                                if (X(i, j, k) > max_val) {
                                    max_val = X(i, j, k);
                                    max_i = i;
                                    max_j = j;
                                }
                            }
                        }
                    }
                    if (max_i != -1 && max_j != -1) {
                        // Propagate the gradient only to the position that attained the max value
                        GradX(max_i, max_j, k) = next->GradX(x, y, k);
                    }
                }
            }
        }
    }
    else {
        std::cerr << "updateGradXFromPooling: Unsupported pooling type." << std::endl;
    }
}

// Helper function: backpropagation for a flatten next layer
void updateGradXFromFlatten(Layer* next, RTensor &GradX, const RTensor &X) {
    for (Integer i = 0; i < GradX.dims(0); ++i) {
        for (Integer j = 0; j < GradX.dims(1); ++j) {
            for (Integer k = 0; k < GradX.dims(2); ++k) {
                Integer p = i * X.dims(1) * X.dims(2) + j * X.dims(2) + k;
                if (next->GradX.rank() != 1) {
                    std::cerr << "updateGradXFromFlatten: GradX of Flatten layer rank is not 1" << std::endl;
                }
                GradX(i, j, k) = next->GradX(p);
            }
        }
    }
}

// Helper function: backpropagation for an activation next layer
void updateGradXFromActivation(Layer* next, RTensor &GradX, const RTensor &X) {
    for (Integer i = 0; i < GradX.dims(0); ++i) {
        for (Integer j = 0; j < GradX.dims(1); ++j) {
            for (Integer k = 0; k < GradX.dims(2); ++k) {
                GradX(i, j, k) = dynamic_cast<Activation*>(next)->dfun_activation(X(i, j, k))
                                  * next->GradX(i, j, k);
            }
        }
    }
}


// ------------------------------ ACTIVATION LAYER ---------------------------  
void Activation::forwardprop() {
    
    Layer* prev = this->prevL(); // Get previous layer
    if (prev->type == _Dense){
        if (typeA == _softMax){
            std :: cout << "try softmax" << std::endl;
            for (Integer i = 0; i < prev->X.size(); ++i) {
                X[i] = utils::softmax(prev->X, i);
            }
        }else{
            for (Integer i = 0; i < prev->X.size(); ++i) {
                X[i] = fun_activation(prev->X[i]);
            }
        }
    }
    else if (prev->type == _Convolution || prev->type == _Pool){
        for (Integer i = 0; i < prev->dims[0]; ++i) {
            for (Integer j = 0; j < prev->dims[1]; ++j) {
                for (Integer k = 0; k < prev->dims[2]; ++k) {
                    //std :: cout << "try" << X(i, j, k) << std::endl;
                    X(i, j, k) = fun_activation(prev->X(i, j, k));
                    //std :: cout << "try"<< std::endl;
                }
            }
        }
    }else{
        std::cerr << "Activation::forwardprop: previous layer is not Dense or Convolution" << std::endl;
    }
}

void Activation::backprop() {
    // if before the last layer (recall that the last layer is the loss layer)
    if (index == network->getLayers().size() - 2) {
        Loss *loss = dynamic_cast<Loss*>(network->getLayers().back()); // the loss is the last layer
        RTensor true_labal = loss->get_vref();
        RTensor pred_labal = this->X;
        GradX = loss->get_dfun_Loss()(true_labal, pred_labal);
        //std:: cout << "first layer in backprop" << std::endl;
    }else{
        if (X.rank() == 1) { // next is also 1D   
            Layer* next = this->nextL(); // get pointer to the next layer, it is a dense layer
            GradX = (next->W).transpose() * next->GradX; 
        }else if (X.rank() == 3) { // next is also 3D
            Layer* next = this->nextL(); // Get pointer to the next layer
    
            switch (next->type) {
                case _Convolution:
                    updateGradXFromConvolution(next, GradX, X);
                    break;
                case _Pool:
                    updateGradXFromPooling(next, GradX, X);
                    break;
                case _Flatten:
                    updateGradXFromFlatten(next, GradX, X);
                    break;
                case _Activation:
                    updateGradXFromActivation(next, GradX, X);
                    break;
                default:
                    std::cerr << "Convolution::backprop: next layer type is not supported." << std::endl;
                    break;
            }

        }else{
            std::cerr << "Activation::backprop: X is not a 1D or 3D tensor" << std::endl;   
        }

    }
}

// ------------------------------ CONVOLUTION LAYER ---------------------------

/**
* @brief Sets X and the kernels given the dimensions of the previous layer.
* 
* @param nb_channels_prev
* @param n_prev
*/

void Convolution::set_dims(Integer nb_channels_prev, Integer n_prev) {
    // e.g. the new spatial dimension after a valid conv (square)
    Integer new_n = n_prev - ker_size + 1;

    // Store these in dims (used for the "output" shape of this layer)
    dims[0] = new_n;
    dims[1] = new_n;
    dims[2] = nb_kers;

    double fan_prev = nb_channels_prev * ker_size * ker_size;
    double fan_out = nb_kers * ker_size * ker_size;
    double limit = std::sqrt(6.0 / (fan_prev + fan_out));

    W = RTensor(ker_size, ker_size, nb_channels_prev, nb_kers);

    // Initialize weights with uniform distribution in range [-limit, +limit]
    for (Integer oc = 0; oc < nb_kers; ++oc) {
        for (Integer ic = 0; ic < nb_channels_prev; ++ic) {
            for (Integer kr = 0; kr < ker_size; ++kr) {
                for (Integer kc = 0; kc < ker_size; ++kc) {
                    double r = ((double)std::rand() / (double)RAND_MAX) * 2 * limit - limit;
                    W(kr, kc, ic, oc) = r;
                }
            }
        }
    }

    // Prepare multi-D X and GradX to hold the activation
    // shape: (new_n, new_n, nb_kers)
    X = RTensor(new_n, new_n, nb_kers);
    GradX= RTensor(new_n, new_n, nb_kers);

    // Prepare multi-D GradW, GradWm with same shape as W 
    GradW = RTensor(ker_size, ker_size, nb_channels_prev, nb_kers);
    GradWm = RTensor(ker_size, ker_size, nb_channels_prev, nb_kers);
}


void Convolution::forwardprop() {
    Layer* prevLayer = this->prevL();
    // prevLayer->dims = (n_prev, n_prev, nb_channels_prev)

    Integer nb_channels_prev = prevLayer->dims[2];
    Integer current_n = dims[0];
    
    for (Integer outChan = 0; outChan < nb_kers; ++outChan) { // for each kernel of the conv layer 
        // We'll accumulate in a 2D matrix the partial sums
        RTensor X_c(current_n, current_n); // this is one channel of the activation

        for (Integer inChan = 0; inChan < nb_channels_prev; ++inChan){ // sum of the channels of the previous layer
            RTensor ker2D = W.slice_2d(inChan, outChan);
            
            // Extract a 2D slice from prevX
            RTensor PrevX_Channel = prevLayer->X.slice_2d(inChan);
            
            // Convolve
            RTensor convRes = utils::convolve(ker2D, PrevX_Channel); // put the kernel first

            // accumulate convRes into X_c
            X_c += convRes;

        }
        // Store X_c into this->X at channel outChan
        this->X.set_channel(outChan, X_c);
    }
}


void Convolution::backprop() {
    // First compute GradX by calling the appropriate helper function based on the next layer type
    Layer* next = this->nextL(); // Get pointer to the next layer
    
    switch (next->type) {
        case _Convolution:
            updateGradXFromConvolution(next, GradX, X);
            break;
        case _Pool:
            updateGradXFromPooling(next, GradX, X);
            break;
        case _Flatten:
            updateGradXFromFlatten(next, GradX, X);
            break;
        case _Activation:
            updateGradXFromActivation(next, GradX, X);
            break;
        default:
            std::cerr << "Convolution::backprop: next layer type is not supported." << std::endl;
            break;
    }
    
    // Now compute GradW (weight gradients)
    Integer current_n = this->dims[0];
    Layer* prev = this->prevL(); // Get pointer to the previous layer
    
    // Assuming W has dimensions: [ker_size, ker_size, in_channels, out_channels]
    for (Integer i = 0; i < W.dims(0); i++) {
        for (Integer j = 0; j < W.dims(1); j++) {
            for (Integer k = 0; k < W.dims(2); k++) {
                for (Integer c = 0; c < W.dims(3); c++) {
                    // Initialize weight gradient to zero
                    GradW(i, j, k, c) = 0;
                    for (Integer u = 0; u < current_n; u++) {
                        for (Integer v = 0; v < current_n; v++) {
                            // prev->X is assumed to be a 3D tensor (for example, from a Conv or Pool layer)
                            GradW(i, j, k, c) += GradX(u, v, c) * prev->X(i + u, j + v, k);
                        }
                    }
                }
            }
        }
    }
}


// -------------------------------- REDUCTION LAYER ---------------------------

void Pool::set_dims(Integer prev_n, Integer prev_c) {
    // e.g. out size = (prev_n - p)/stride + 1 if "valid" pooling
    Integer new_n = (prev_n - p) / stride + 1;

    dims[0] = new_n;
    dims[1] = new_n;
    dims[2] = prev_c;

    X = RTensor(new_n, new_n, prev_c);
    GradX = RTensor(new_n, new_n, prev_c);
}

void Pool::forwardprop() {

    Layer* prev = this->prevL(); // Get previous layer
    Integer nb_channels = dims[2];
    Integer current_n = dims[0];
    // Boucle sur chaque canal de la carte de caractéristiques
    for (Integer c = 0; c < nb_channels; ++c) {
        // Boucle sur la hauteur de la carte
        for (Integer i = 0; i < current_n; ++i) {
            // Boucle sur la largeur de la carte
            for (Integer j = 0; j < current_n; ++j) {
                // Initialisation pour max pooling ou mean pooling
                if (typeP == _maxPool) {
                    float max_val = std::numeric_limits<float>::lowest();
                    for (Integer u = 0; u < p; ++u) {
                        for (Integer v = 0; v < p; ++v) {
                            Integer x = stride * i + u;
                            Integer y = stride * j + v;
                            max_val = std::max(max_val, prev->X(x, y, c));
                        }
                    }
                    X(i, j, c) = max_val;
                } else if (typeP == _meanPool) {
                    float sum = 0.0f;
                    for (Integer u = 0; u < p; ++u) {
                        for (Integer v = 0; v < p; ++v) {
                            Integer x = stride * i + u;
                            Integer y = stride * j + v;
                            sum += prev->X(x, y, c);
                        }
                    }
                    X(i, j, c) = sum / (p * p);
                }
            }
        }
    }
}

void Pool::backprop() {
    Layer* next = this->nextL(); // Get pointer to the next layer
    
    switch(next->type) {
        case _Convolution:
            updateGradXFromConvolution(next, GradX, X);
            break;
            
        case _Pool:
            updateGradXFromPooling(next, GradX, X);
            break;
            
        case _Flatten:
            updateGradXFromFlatten(next, GradX, X);
            break;
            
        case _Activation:
            updateGradXFromActivation(next, GradX, X);
            break;
            
        default:
            std::cerr << "Pool::backprop: next layer type is not supported." << std::endl;
            break;
    }
}

// ------------------------------ FLATTEN LAYER ---------------------------

void Flatten::backprop() {

    // if before the last layer (recall that the last layer is the loss layer)
    if (index == network->getLayers().size() - 2) {
        Loss *loss = dynamic_cast<Loss*>(network->getLayers().back()); // the loss is the last layer
        RTensor true_labal = loss->get_vref();
        RTensor pred_labal = this->X;
        GradX = loss->get_dfun_Loss()(true_labal, pred_labal);
        //std:: cout << "first layer in backprop" << std::endl;
    }else{
        Layer* next = this->nextL(); // get pointer to the next layer
        if (next->type == _Dense){
            //std :: cout << "get dX_dPrevX_Dense Dense" << std::endl;
            GradX = dynamic_cast<Dense*>(next)->get_dX_dPrevX_Dense() * next->GradX; // only update the GradX
        }
        else if (next->type == _Activation){
            //std :: cout << "get dX_dPrevX_Dense Actv" << std::endl;
            GradX = dynamic_cast<Activation*>(next)->get_dX_dPrevX_Dense() * next->GradX; // only update the GradX
        } else{
            std::cerr << "Flatten::backprop: next layer is not Dense or Activation" << std::endl;
        }
       
    }
}

void Flatten::forwardprop() {
    Layer* prev = this->prevL(); // Get previous layer
    Integer prev_d1 = prev->dims[0]; // Get the size of the previous layer
    Integer prev_d2 = prev->dims[1]; 
    Integer prev_d3 = prev->dims[2]; // Get the number of channels of the previous layer
    Integer k = 0;
    for (Integer i = 0; i < prev_d1; ++i) {
        for (Integer j = 0; j < prev_d2; ++j) {
            for (Integer l = 0; l < prev_d3; ++l) {
                X(k) = prev->X(i, j, l);
                k++;
            }
        }
    }
}


//-------------------------------- NETWORK METHODS ----------------------------


void Network::add(Layer* layer) { 
    if (layer->type == _Dense){
        Integer n_prev = layers.back()->dims[0]; //get the size of the last layer of the network, for connection
        // note that Dense can only be applied if previous layer is flat so the dims are (n, 1, 1)
        dynamic_cast<Dense*>(layer)->set_dims(n_prev); // Cast the layer to Dense and set its parameters
    }
    else if (layer->type == _Activation){ // copy the dimensions of the previous layer
        Layer* prev = layers.back(); // Get the previous layer
        Integer prev_d1 = prev->dims[0]; // Get the size of the previous layer
        Integer prev_d2 = prev->dims[1];
        Integer prev_d3 = prev->dims[2]; // Get the number of channels of the previous layer
        TypeLayer prev_type = prev->type;
        dynamic_cast<Activation*>(layer)->set_dims(prev_d1, prev_d2, prev_d3, prev_type); // Cast the layer to Dense and set its parameters
    }
    else if (layer->type == _Convolution){
        Integer nb_channels_prev = layers.back()->dims[2]; //get the number of channels of the last layer of the network, for connection
        Integer n_prev = layers.back()->dims[0]; //get the size of the last layer of the network, for connection
        dynamic_cast<Convolution*>(layer)->set_dims(nb_channels_prev, n_prev); // Cast the layer to Convolution and set its parameters
        
    }
    else if (layer->type == _Pool){
        Layer* prev = layers.back(); // Get the previous layer
        Integer prev_n = prev->dims[0]; // Get the size of the previous layer
        Integer prev_c = prev->dims[2]; // Get the number of channels of the previous layer
        dynamic_cast<Pool*>(layer)->set_dims(prev_n, prev_c); 
    } 
    else if (layer->type == _Flatten){
        Layer* prev = layers.back(); // Get the previous layer
        Integer prev_d1 = prev->dims[0]; // Get the size of the previous layer
        Integer prev_d2 = prev->dims[1]; 
        Integer prev_d3 = prev->dims[2]; // Get the number of channels of the previous layer
        dynamic_cast<Flatten*>(layer)->set_dims(prev_d1, prev_d2, prev_d3);
    }
    layers.push_back(layer); 
    layer->network = this;
    layer->index = layers.size() - 1;
    //std :: cout << "layer added" << std::endl;
}


void Network::forwardprop(const RTensor& x_i, const RTensor& y_i) {
    if (layers.empty()) {
        cerr << "Error: No layers in the network!\n";
        return;
    }

    layers[0]->X = x_i; // Set input as the activation of the first layer

    Layer* last = layers[layers.size() - 1]; // Get last layer
    Loss* loss_layer = dynamic_cast<Loss*>(last); // Ensure it's a loss layer
    if (loss_layer) {
        loss_layer->set_vref(y_i); // Set the reference value for the loss layer
    } else {
        cerr << "Error: Last layer is not a Loss layer!\n";
    }

    // Forward propagation through all layers, starting from the second layer
    for (size_t i = 1; i < layers.size(); ++i) {
        //std :: cout << "forwardprop layer " << i << " of type " << layers[i]->type << std::endl;
        layers[i]->forwardprop();
    }
}

void Network::backprop() {
    // Ensure there are layers
    if (layers.empty()) {
        cerr << "Error: No layers in network!\n";
        return;
    }

    // Backpropagate in reverse order
    for (int i = layers.size() - 1; i >= 0; --i) {
        //std :: cout << "backprop layer " << i << " of type " << layers[i]->type << std::endl;
        layers[i]->backprop();
    }
}

void Network::majparametres(TypeStep tp, Reel rho, Reel alpha, Integer k) {
    for (Layer* layer : layers) {
        if (layer->params) layer->majparameters(tp, rho, alpha, k);
    }
}


/** 
 * @brief Splits the data into batches
 *
 * @param Es The input data (features)
 * @param Ss The output data (labels)
 *
 * @return A vector of batches (where a batch is a vector of 2 vectors, one with the Es, and one with the Ss)
 */
 vector<vector<vector<RTensor>>> Network::batch_split(const vector<RTensor>& Es, const vector<RTensor>& Ss) {
    vector<RTensor> shuffled_Es = Es; // Copy the input data
    vector<RTensor> shuffled_Ss = Ss; // Copy the output data

    // Shuffle the data
    utils::shuffle_data(shuffled_Es, shuffled_Ss);  

    // Determine batch size
    size_t total_samples = shuffled_Es.size();
    size_t batch_size_ = (batch_size > 0) ? batch_size : total_samples; // Use full dataset if batch_size is invalid

    vector<vector<vector<RTensor>>> batches; // Container for batches

    // Create batches
    for (size_t i = 0; i < total_samples; i += batch_size_) {
        vector<RTensor> batch_Es, batch_Ss;

        for (size_t j = 0; j < batch_size_ && (i + j) < total_samples; ++j) {
            batch_Es.push_back(shuffled_Es[i + j]);
            batch_Ss.push_back(shuffled_Ss[i + j]);
        }

        batches.push_back({batch_Es, batch_Ss});
    }

    return batches;
}


/** @brief Computes the gradient over a batch of data
 * 
 * @param batch_Es The input data (features)
 * @param batch_Ss The output data (labels)
 */
void Network::train_batch(const vector<RTensor>& batch_Es, const vector<RTensor>& batch_Ss) {
    for (size_t i = 0; i < batch_Es.size(); ++i) {
        forwardprop(batch_Es[i], batch_Ss[i]);
        backprop(); // update the gradient
    }
}

/** @brief Trains the network using the given data
 * 
 *  @param Es The input data (features)
 *  @param Ss The output data (labels)
 *  @param tp The type of step (e.g., fixed, linear, quadratic, exponential)
 *  @param rho The learning rate
 *  @param alpha The momentum factor
 */
 void Network::train(const vector<RTensor>& Es, const vector<RTensor>& Ss, TypeStep tp, Reel rho, Reel alpha) {
    
    Integer iteration = 0; // Tracks the total number of times `majParametres` is called

    for (Integer epoch = 0; epoch < epochs; ++epoch) {
        cout << "Epoch " << (epoch + 1) << "/" << epochs << "...\n"; // Print progress

        vector<vector<vector<RTensor>>> batches = batch_split(Es, Ss);

        // Iterate through batches
        for (size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx) {
            train_batch(batches[batch_idx][0], batches[batch_idx][1]);
            // now that the sum of the gradients is done, take the average 
            // for each layer with weights, divide GradWm by the batch size
            for (Layer* layer : layers) {
                if (layer->params) {
                    layer->GradWm *= (1.0 / static_cast<Reel>(batch_size)); // will work for any size
                }
            }
            majparametres(tp, rho, alpha, iteration); // Use iteration counter
            iteration++; // Increment total iteration count
        }
    }
}

void Network::print(ostream& out) const {
    out << "Network: " << name << "\n";
    for (Layer* layer : layers) {
        layer->print(out);
    }
}

void Network::test(const vector<RTensor>& Es, const vector<RTensor>& Ss) {
    for (size_t i = 0; i < Es.size(); ++i) {
        forwardprop(Es[i], Ss[i]);
        // recall : last layer is the loss layer
        // the prediction is the activation of second to last layer
        cout << "Prediction: " << layers[layers.size() - 2]->X << ", Actual: " << Ss[i] << endl;
    }
}