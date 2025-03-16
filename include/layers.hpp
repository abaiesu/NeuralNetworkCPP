#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <iostream>
#include <vector>
#include "basic_data.hpp"
#include "helpers.hpp"
#include "decl.hpp"

using namespace std;

class Network; // Forward declaration

class Layer {
public:
    Network* network = nullptr;
    TypeLayer type;
    Integer dims[3] = {0, 0, 0};
    Integer index = 0;
    RVector X;
    RVector GradX; // dX/dPrev
    RMatrix GradW; // dX/dW
    RMatrix GradWm; // (Optional: Accumulated/averaged gradient over multiple samples)


    
    // constructor
    Layer() = default;
    Layer(TypeLayer t) : type(t) {}
    Layer(TypeLayer t, Integer d1, Integer d2, Integer d3) : type(t)  {
        dims[0] = d1;
        dims[1] = d2;
        dims[2] = d3;
        X = RVector(d1);
        GradX = RVector(d1);
    }

    bool params = false; // true if the layer has parameters (e.g. yes for dense, no for activation)
    bool flagP = false;
    bool initGradPm = true;

    virtual ~Layer() = default;
    virtual Layer* clone() const = 0;
    virtual void forwardprop() {} // update X
    virtual void backprop() {} // update GradX, GradW
    virtual void majparameters(TypeStep tp, Reel rho, Reel alpha, Integer k) {} // gradient iteration
    void print(std::ostream& out) const;
    virtual RMatrix dXj_dPrevi() {}
    Layer* nextL(); //get the next layer
    Layer* prevL(); //get the previous layer
};


class Network {
    protected:
        vector<Layer*> layers; // List of layer pointers
        string name = ""; // Output file name
        RVector residuals; // Residuals vector
        Integer batch_size = -1; // -1 = no mini-batch
        Integer epochs = 10; // Number of training epochs
    public:
        // Constructors
        Network() = default;
        Network(const string& name, Integer b, Integer epochs) : name(name), batch_size(b), epochs(epochs){}

        // Destructor
        // Destructor
        ~Network() {
            for (Layer* layer : layers) {
                delete layer; // Delete each layer
            }
        }
        // Getter for layers
        const vector<Layer*>& getLayers() const { return layers; }
    
        // Add layer to network
        void add(Layer* layer) ;
        // Forward and Backpropagation
        void forwardprop(const RVector& E, const RVector& S = RVector());
        void backprop();
    
        // Update Parameters
        void majparametres(TypeStep tp, Reel rho, Reel alpha, Integer k);
    
        // Training Methods
        vector<vector<vector<RVector>>> batch_split(const vector<RVector>& Es, 
                                                            const vector<RVector>& Ss);
        void train_batch(const vector<RVector>& Es, const vector<RVector>& Ss);
        void train(const vector<RVector>& Es, const vector<RVector>& Ss, TypeStep tp, 
                    Reel rho = 0.01, Reel alpha = 0.001);
    
        // Test Network
        void test(const vector<RVector>& Es, const vector<RVector>& Ss);
        
        // Print
        void print(ostream& out) const;
    };


class Entry : public Layer {
public:
    // constructor
    Entry(Integer n) : Layer(_Entry, n, 1, 1) {}
    Entry* clone() const override { return new Entry(*this); }
    ~Entry() = default;
};

class Convolution : public Layer {
protected:
    RMatrix K; // Convolution kernel
    Integer mu = 1, nu = 1; // Strides
    Integer i0 = 1, j0 = 1; // Start indices
    bool same_size = false; // Preserve original size (padding)

public:
    void randomK(Integer p, Integer q = 0);
    Convolution* clone() const override { return new Convolution(*this); }
    void forwardprop() override;
    void backprop() override;
    void majparameters(TypeStep tp, Reel rho, Reel alpha, Integer k) override;
    ~Convolution() = default;
    //void print(std::ostream& out) const override;
};

class Reduction : public Layer {
protected:
    TypeReduction typeR;
    Integer p, q;

public:
    Reduction* clone() const override;
    void forwardprop() override;
    void backprop() override;
    ~Reduction() = default;
    //void print(std::ostream& out) const override {}
};

class Activation : public Layer {
protected:
    TypeActivation typeA;
    FR_p fun_activation;
    FR_p dfun_activation;

public:
    // constructor
    Activation(TypeActivation t) : Layer(_Activation), typeA(t) {
        fun_activation = (t == _relu) ? utils::relu : (t == _tanh) ? utils::hyper_tan : (t == _abs_tanh) ? utils::abs_hyper_tan : utils::sigmoid;
        dfun_activation = (t == _relu) ? utils::d_relu : (t == _tanh) ? utils::d_hyper_tan : (t == _abs_tanh) ? utils::d_abs_hyper_tan : utils::d_sigmoid;
    }
    Activation* clone() const override { return new Activation(*this); }
    void forwardprop() override;
    void backprop() override;
    void majparameters(TypeStep tp, Reel rho, Reel alpha, Integer k) override {
        (void)tp;
        (void)rho;
        (void)alpha;
        (void)k;
    }; // no majparameters because no parameters
    virtual RMatrix dXj_dPrevi() ;
    ~Activation() = default;
    //void print(std::ostream& out) const override {}
};

class Dense : public Layer {
protected:
    RMatrix W;
public:
    // constructor
    Dense(Integer n) : Layer(_Dense, n, 1, 1) {
        params = true;
    };
    RMatrix get_W() const { return W; } 
    void set_params(Integer prev_size);
    Dense* clone() const override;
    void forwardprop() override;
    void backprop() override;
    void majparameters(TypeStep tp, Reel rho, Reel alpha, Integer k) override;
    virtual RMatrix dXj_dPrevi() { return W.transpose(); }
    //void print(std::ostream& out) const override {}
    ~Dense() = default;
};

class Loss : public Layer{
protected:
    TypeLoss typeP;
    FV2_p fun_Loss;
    FV2V_p dfun_Loss;
    RVector vref;
public:
    // constructor
    Loss(TypeLoss t1) : Layer(_Loss, 1, 1, 1), typeP(t1) { // X is just 1 number
        setFunPtr();
        params = false;  // Initialize function pointers
    }
    RVector get_vref() const { return vref; }
    void set_vref(const RVector& v) { vref = v; }
    FV2_p get_fun_Loss() const { return fun_Loss; }
    FV2V_p get_dfun_Loss() const { return dfun_Loss; }
    void setFunPtr();
    Loss* clone() const override { return new Loss(*this); }
    void forwardprop() override;
    void backprop() override {} // no backprop because no parameters
    void majparameters(TypeStep tp, Reel rho, Reel alpha, Integer k) override {
        (void)tp;
        (void)rho;
        (void)alpha;
        (void)k;
    }; // no majparameters because no parameters
    //virtual void print(std::ostream& out) const;
    virtual ~Loss() = default;
};

#endif // LAYERS_H
