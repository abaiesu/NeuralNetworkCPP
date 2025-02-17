#ifndef LAYERS_H
#define LAYERS_H

#include <iostream>
#include <vector>
#include "basic_data.h"


enum TypeLayer { _nondefini, _Entry, _convolution, _reduction, _activation, _Dense, _Loss };
enum TypeReduction { _maxReduction, _moyenneReduction };
enum TypeActivation { _activation_indefini, _relu, _tanh, _tanhsat, _sigmoide};
enum TypeLoss { _moindre_carre, _moindre_abs, _huber, _entropie_croisee, _softMax};
enum TypeStep {_fixed, _linear, _quadratic, _exponential};

typedef Reel (*FR_p)(Reel); // Function pointer for activation functions
typedef Reel (*FV2_p)(const std::vector<Reel>&, const std::vector<Reel>&); // Function pointer for loss functions
typedef std::vector<Reel> (*FV2V_p)(const std::vector<Reel>&, const std::vector<Reel>&); // Function pointer for loss gradients

class Network; // Forward declaration

class Layer {
public:
    Network* network = nullptr;
    TypeLayer type;
    Integer dims[3] = {0, 0, 0};
    Integer index = 0;
    RVector X;  // X_i = sum_j (w_ij * prev_j) 
    // This represents the activations of the neurons in the current layer.
    // Each element X_i is computed as a weighted sum of the previous layer's activations (prev_j)
    // using the weight matrix w_ij.
    // X = W * Prev 

    RVector GradX;  // GradX_i = dL/dX_i = sum_j (w_ij * dL/dprev_j)
    // This represents the gradient of the loss function L with respect to X.
    // It is computed using the chain rule: GradX propagates the gradient from the next layer.
    // GradX = W^T * GradPrev  

    RMatrix GradP;  // GradP_ij = dL/dW_ij = prev_j * GradX_i
    // This represents the gradient of the loss with respect to the weight parameters W_ij.
    // Using the chain rule, the derivative of L with respect to W_ij is given by:
    // GradP_ij = prev_j * GradX_i  
    // This is needed for updating the weights during backpropagation.

    RMatrix GradPm; // (Optional: Accumulated/averaged gradient over multiple samples)

    bool parameters = false; // true if the layer has parameters (e.g. yes for dense, no for activation)
    bool flagP = false;
    bool initGradPm = true;

    virtual ~Layer() = default;
    virtual Layer* clone() const = 0;
    virtual void forwardprop() {} // update X
    virtual void backprop() {} // update GradX
    virtual void majparameters(TypeStep tp, Reel rho, Reel alpha, Integer k) {} // gradient iteration
    virtual void print(std::ostream& out) const {}

    Layer* nextL(); 
    Layer* prevL();
};

class Entry : public Layer {
public:
    Entry* clone() const override { return new Entry(*this); }
    void print(std::ostream& out) const override {}
};

class Convolution : public Layer {
protected:
    std::vector<std::vector<Reel>> K; // Convolution kernel
    Integer mu = 1, nu = 1; // Strides
    Integer i0 = 1, j0 = 1; // Start indices
    bool same_size = false; // Preserve original size (padding)

public:
    void randomK(Integer p, Integer q = 0);
    Convolution* clone() const override { return new Convolution(*this); }
    void prop() override;
    void backprop() override;
    void majparameters(int tp, Reel rho, Reel alpha, Integer k) override;
    void print(std::ostream& out) const override;
};

class Reduction : public Layer {
protected:
    TypeReduction typeR;
    Integer p, q;

public:
    Reduction* clone() const override;
    void forwardprop() override;
    void backprop() override;
    void print(std::ostream& out) const override {}
};

class Activation : public Layer {
protected:
    TypeActivation typeA;
    FR_p fun_activation;
    FR_p dfun_activation;

public:
    Activation* clone() const override;
    void forwardprop() override;
    void backprop() override;
    void majparameters(int tp, Reel rho, Reel alpha, Integer k) override;
    void print(std::ostream& out) const override {}
};

class Dense : public Layer {
protected:
    RMatrix C;

public:
    Dense* clone() const override;
    void forwardprop() override;
    void backprop() override;
    void majparameters(int tp, Reel rho, Reel alpha, Integer k) override;
    void print(std::ostream& out) const override {}
};

class Loss {
protected:
    TypeLoss
 typeP;
    FV2_p fun_Loss;
    FV2V_p dfun_Loss;
    std::vector<Reel> vref;

public:
    void setFunPtr();
    virtual Loss* clone() const;
    virtual void forwardprop();
    virtual void backprop();
    virtual void print(std::ostream& out) const;
};

#endif // LAYERS_H
