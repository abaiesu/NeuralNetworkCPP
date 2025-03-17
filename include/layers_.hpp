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
    
    RVector X;          // 1D data, but not used for conv/pool
    RTensor X_tensor;   // The actual multi-D data for conv/pool

    RVector GradX;      // 1D gradient
    RTensor GradX_tensor; // The multi-D gradient

    RMatrix W;             
    RMatrix GradW;
    RMatrix GradWm;

    // and for convolution weights:

    RTensor W_tensor;        // e.g. (ker_size, ker_size, in_channels, out_channels)
    RTensor GradW_tensor;    // etc.
    RTensor GradWm_tensor;


    
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
    virtual void majparameters(TypeStep tp, Reel rho, Reel alpha, Integer k); // gradient iteration
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
    Integer nb_kers;
    Integer ker_size;

    //Integer mu = 1, nu = 1; // Strides
    //Integer i0 = 0, j0 = 0; // Start indices
    //bool same_size = false; // Preserve original size (padding)

public:

    //void randomK(Integer p, Integer q = 0);
    Convolution(Integer nb_kers, Integer ker_size) : Layer(_Convolution) {
        params = true;
        this->nb_kers = nb_kers;
        this->ker_size = ker_size;
    }
    Convolution* clone() const override { return new Convolution(*this); }
    void set_params(Integer prev_channels, Integer prev_n);
    void forwardprop() override;
    void backprop() override;
    ~Convolution() = default;
    //void print(std::ostream& out) const override;
};

class Pool : public Layer {
protected:
    Integer p;
    Integer stride;
public:
    TypePool typeP;
    Pool* clone() const override { return new Pool(*this); }
    Pool (TypePool t, Integer p, Integer stride) : Layer(_Pool), typeP(t), p(p), stride(stride) {}
    void set_params(Integer prev_n, Integer prev_c);
    void forwardprop() override;
    void backprop() override{}; // no backpropagation since no params
    Reel get_dX_dPrevX() { return 1.0 / (p * p); }
    ~Pool() = default;
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
    RMatrix dXj_dPrevi() ;
    Reel get_dX_dPrevX(Integer i, Integer j, Integer k, Integer i_p, Integer j_p, Integer c){
        if (i == i_p && j == j_p && k == c){
            return dfun_activation(X_tensor(i, j, k));
        }
        return 0;
    }
    ~Activation() = default;
    //void print(std::ostream& out) const override {}
};

class Dense : public Layer {
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
    //virtual void print(std::ostream& out) const;
    virtual ~Loss() = default;
};


class Flatten : public Layer {
    public:
        // constructor
        Flatten() : Layer(_Flatten) {};
        void set_params(Integer d1, Integer d2, Integer d3){
            dims[0] = d1 * d2 * d3;
            dims[1] = 1;
            dims[2] = 1;
            X = RVector(d1 * d2 * d3);
            GradX = RVector(d1 * d2 * d3);
        }
        Reel get_dX_dPrevX(Integer p, Integer i, Integer j, Integer k, Integer prev_n){
            if (p == i*prev_n*prev_n + j*prev_n + k){
                return 1;
            }
            return 0;
        }
        Flatten* clone() const override { return new Flatten(*this); }
        void forwardprop() override {}
        void backprop() override {}
        ~Flatten() = default;
    };

#endif // LAYERS_H
