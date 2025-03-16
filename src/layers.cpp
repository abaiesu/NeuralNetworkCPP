#include "layers.hpp"
#include "helpers.hpp"
#include <random> 

//------------------------------ GENERIC LAYER --------------------------------

Layer* Layer::nextL() {
    // if the layer is the last one or the network is not set, return nullptr
    if (network == nullptr || index >= network->getLayers().size() - 1) {
        std::cerr << "Layer::nextL: no next layer found." << std::endl;
        return nullptr;
    }
    return network->getLayers()[index + 1];
}

Layer* Layer::prevL() {
    // if the layer is the first one or the network is not set, return nullptr
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


// ------------------------------ DENSE LAYER ---------------------------------


void Dense::set_params(Integer n_prev) {

    Integer n = dims[0];
    W = RMatrix(n, n_prev); // from a space of size n_prev to a space of size n

    // Compute Glorot/Xavier uniform limit
    double limit = std::sqrt(6.0 / (n + n_prev));

    // Random initialization in [-limit, limit]
    for (Integer i = 0; i < n; ++i) {
        for (Integer j = 0; j < n_prev; ++j) {
            double r = (double)std::rand() / (double)RAND_MAX; // in [0,1]
            W(i, j) = r * 2.0 * limit - limit; // in [-limit, limit]
        }
    }

    GradW = RMatrix(n, n_prev); // Empty matrix for gradient
    GradWm = RMatrix(n, n_prev); // Empty matrix for average gradient
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
    //std :: cout << "ok forwardprop dense" << std::endl;
}

/**
 * @brief Performs backward propagation for the Dense layer.
 * 
 * 1. Propagates the error backward: previous->GradX += C^T * GradX.
 * 2. Computes the gradient of the weights: GradC = outer_product(GradX, previous->X)
*/
void Dense::backprop() {

    //std::cout << "backprop DENSE" << std::endl;
    // Compute gradX 

    // if before the last layer (recall that the last layer is the loss layer)
    if (index == network->getLayers().size() - 2) {
        Loss *loss = dynamic_cast<Loss*>(network->getLayers().back()); // the loss is the last layer
        RVector true_labal = loss->get_vref();
        RVector pred_labal = this->X;
        GradX = loss->get_dfun_Loss()(true_labal, pred_labal);
        //std:: cout << "first layer in backprop" << std::endl;
    }else{
        //std :: cout << "get grad X" << std::endl;
        Layer* next = this->nextL(); // get pointer to the next layer
        RMatrix next_X_vs_PrevX = next->dXj_dPrevi(); 
        GradX = next_X_vs_PrevX * next->GradX; // only update the GradX
        //std:: cout << "not first layer in backprop" << std::endl;
    }

    //std :: cout << "hi ";
    // Compute gradW and gradWm
    Layer* prev = this->prevL(); // get pointer to the previous layer
    //std :: cout << "vecteur prev->X: " << prev->X.size() << std::endl;
    //std :: cout << "vecteur GradX: " << GradX.size() << std::endl;
    GradW = utils::outerProduct(GradX, prev->X);
    //std :: cout << "GradW: ";
    //GradW.print_size();
    //std :: cout << "GradWm: ";
    //GradWm.print_size();
    //std :: cout << "W";
    //W.print_size();
    GradWm += GradW; 

    //std::cout << "GradWm: " << GradWm << std::endl;
    //std::cout << "\n" << std::endl;
}


/**
 * @brief Updates the weights of the Dense layer for 1 iteration of gradient descent
 * 
 * @param tp The type of step (e.g., fixed, linear, quadratic, exponential).
 * @param rho The learning rate.
 * @param alpha The momentum factor.
 * @param k The current iteration number.
 */
 void Dense::majparameters(TypeStep tp, Reel rho, Reel alpha, Integer k) {
    
    Reel step = rho; //utils::computeLearningRate(tp, rho, alpha, k);
    Reel v = 0;
    GradWm *= step; // update the gradient
    W -= GradWm; // update the weights
    GradWm = RMatrix(GradWm.rows, GradWm.cols); // reset the gradient
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
        default:
            throw std::invalid_argument("Invalid loss type");
    }
}


void Loss::forwardprop() {
    //std:: cout << "in loss forwardprop" << std::endl;
    Layer* prev = this->prevL(); // Get previous layer
    //std :: cout << "prev is of type: " << prev->type << std::endl;
    RVector true_labal = vref;
    RVector pred_labal = prev->X;
    this->X = RVector(1, (*fun_Loss)(true_labal, pred_labal));
    //std :: cout << "prev->X: " << prev->X << " of size: " << prev->X.size() << std::endl;
    //std :: cout << "vref: " << vref << std::endl;
    //std :: cout << "X: " << X << "\n" << std::endl;
}

// ------------------------------ ACTIVATION LAYER ---------------------------  

void Activation::forwardprop() {
    Layer* prev = this->prevL(); // Get previous layer
    for (Integer i = 0; i < prev->X.size(); ++i) {
        X[i] = fun_activation(prev->X[i]);
    }
}

void Activation::backprop() {
    // if before the last layer (recall that the last layer is the loss layer)
    if (index == network->getLayers().size() - 2) {
        Loss *loss = dynamic_cast<Loss*>(network->getLayers().back()); // the loss is the last layer
        RVector true_labal = loss->get_vref();
        RVector pred_labal = this->X;
        GradX = loss->get_dfun_Loss()(true_labal, pred_labal);
        //std:: cout << "first layer in backprop" << std::endl;
    }else{
        Layer* prev = this->prevL(); // Get previous layer
        RVector d_f_X_prev = RVector(prev->X.size());
        for (Integer i = 0; i < prev->X.size(); ++i) {
            d_f_X_prev[i] = dfun_activation(prev->X[i]);
        }
        GradX = d_f_X_prev * prev->GradX; // only update the GradX
    }
}

RMatrix Activation::dXj_dPrevi() {
    Layer* prev = this->prevL(); // Get previous layer
    RMatrix d_f_X_prev = RMatrix(X.size(), X.size());
    for (Integer i = 0; i < X.size(); ++i) {
        d_f_X_prev(i, i) = dfun_activation(prev->X[i]);
    }
    return d_f_X_prev;
}

//-------------------------------- NETWORK METHODS ----------------------------


void Network::add(Layer* layer) { 
    if (layer->type == _Dense){
        Integer n_prev = layers.back()->dims[0]; //get the size of the last layer of the network, for connection
        // note that Dense can only be applied if previous layer is flat so the dims are (n, 1, 1)
        dynamic_cast<Dense*>(layer)->set_params(n_prev); // Cast the layer to Dense and set its parameters
    }
    if (layer->type == _Activation){ // copy the dimensions of the previous layer
        Layer* prev = layers.back(); // Get the previous layer
        std::copy(std::begin(prev->dims), std::end(prev->dims), std::begin(layer->dims));
        layer->X = RVector(prev->X.size()); // Initialize X with the same size as the previous layer
    }
    layers.push_back(layer); 
    layer->network = this;
    layer->index = layers.size() - 1;
}


void Network::forwardprop(const RVector& x_i, const RVector& y_i) {
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


/* @brief Splits the data into batches
 * 
 * @param Es The input data (features)
 * @param Ss The output data (labels)
 *
 * @return A vector of batches (where a batch is a vector of 2 vectors, one with the Es, and one with the Ss)
 */
 vector<vector<vector<RVector>>> Network::batch_split(const vector<RVector>& Es, const vector<RVector>& Ss) {
    vector<RVector> shuffled_Es = Es; // Copy the input data
    vector<RVector> shuffled_Ss = Ss; // Copy the output data

    // Shuffle the data
    utils::shuffle_data(shuffled_Es, shuffled_Ss);  

    // Determine batch size
    size_t total_samples = shuffled_Es.size();
    size_t batch_size_ = (batch_size > 0) ? batch_size : total_samples; // Use full dataset if batch_size is invalid

    vector<vector<vector<RVector>>> batches; // Container for batches

    // vector<RVector> batch_Es, batch_Ss;
    // vector<vector<RVector>> batch;
    // vector<vector<vector<RVector>>> batches;

    // Create batches
    for (size_t i = 0; i < total_samples; i += batch_size_) {
        vector<RVector> batch_Es, batch_Ss;

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
void Network::train_batch(const vector<RVector>& batch_Es, const vector<RVector>& batch_Ss) {
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
 void Network::train(const vector<RVector>& Es, const vector<RVector>& Ss, TypeStep tp, Reel rho, Reel alpha) {
    
    Integer iteration = 0; // Tracks the total number of times `majParametres` is called

    for (Integer epoch = 0; epoch < epochs; ++epoch) {
        cout << "Epoch " << (epoch + 1) << "/" << epochs << "...\n"; // Print progress

        vector<vector<vector<RVector>>> batches = batch_split(Es, Ss);

        // Iterate through batches
        for (size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx) {
            train_batch(batches[batch_idx][0], batches[batch_idx][1]);
            // now that the sum of the gradients is done, take the average 
            // for each layer with weights, divide GradWm by the batch size
            for (Layer* layer : layers) {
                if (layer->params) {
                    layer->GradWm *= (1.0 / static_cast<double>(batch_size));
                }
            }
            majparametres(tp, rho, alpha, iteration); // Use iteration counter
            iteration++; // Increment total iteration count
            //std::cout << "batch " << batch_idx << " success" << std::endl;
        }
    }
}

void Network::print(ostream& out) const {
    out << "Network: " << name << "\n";
    for (Layer* layer : layers) {
        layer->print(out);
    }
}

void Network::test(const vector<RVector>& Es, const vector<RVector>& Ss) {
    for (size_t i = 0; i < Es.size(); ++i) {
        forwardprop(Es[i], Ss[i]);
        // recall : last layer is the loss layer
        // the prediction is the activation of second to last layer
        cout << "Prediction: " << layers[layers.size() - 2]->X << ", Actual: " << Ss[i] << endl;
    }
}