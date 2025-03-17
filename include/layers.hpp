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

    RTensor X;     // activation of the neurons
    RTensor GradX; // gradient of J wrt X
 

    // only for layers with params (e.g. dense, convolution)
    RTensor W;     
    RTensor GradW; 
    RTensor GradWm;

    
    // constructor
    Layer() = default;
    Layer(TypeLayer t) : type(t) {}
    Layer(TypeLayer t, Integer d1, Integer d2, Integer d3) : type(t)  {
        dims[0] = d1;
        dims[1] = d2;
        dims[2] = d3;
        if (type == _Dense || type == _Convolution){
            params = true;
        }
        // do not init X and GradX
        // depending on the type of layer, X and GradX will be of rank 1 or 3
    }

    bool params = false; // true if the layer has parameters (e.g. yes for dense, no for activation)
    //bool flagP = false;
    //bool initGradPm = true;

    virtual ~Layer() = default;
    virtual Layer* clone() const = 0;
    virtual void forwardprop() {} // update X
    virtual void backprop() {} // update GradX, GradW
    virtual void majparameters(TypeStep tp, Reel rho, Reel alpha, Integer k); // gradient iteration
    void print(std::ostream& out) const;

    //virtual RTensor get_dX_dPrevX_Dense() {}
    //virtual Reel get_dX_dPrevX_Conv () {}
    //virtual RTensor dXj_dPrevi() {}
    Layer* nextL(); //get the next layer
    Layer* prevL(); //get the previous layer
};


class Network {
    protected:
        vector<Layer*> layers; // List of layer pointers
        string name = ""; // Output file name
        //RTensor residuals; // Residuals vector
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
        void forwardprop(const RTensor& E, const RTensor& S = RTensor());
        void backprop();
    
        // Update Parameters
        void majparametres(TypeStep tp, Reel rho, Reel alpha, Integer k);
    
        // Training Methods
        vector<vector<vector<RTensor>>> batch_split(const vector<RTensor>& Es, 
                                                            const vector<RTensor>& Ss);
        void train_batch(const vector<RTensor>& Es, const vector<RTensor>& Ss);
        void train(const vector<RTensor>& Es, const vector<RTensor>& Ss, TypeStep tp, 
                    Reel rho = 0.01, Reel alpha = 0.001);
    
        // Test Network
        void test(const vector<RTensor>& Es, const vector<RTensor>& Ss);
        
        // Print
        void print(ostream& out) const;
    };


class Entry : public Layer {
public:
    // constructor
    Entry(Integer n) : Layer(_Entry, n, 1, 1) {
        X = RTensor(n);
        GradX = RTensor(n);
    }
    Entry(Integer n1, Integer n2, Integer n3) : Layer(_Entry, n1, n2, n3) { // lorsque l'entrée est une image
        X = RTensor(n1, n2, n3);
        GradX = RTensor(n1, n2, n3);
    }
    Entry* clone() const override { return new Entry(*this); }
    ~Entry() = default;
};

class Convolution : public Layer {
//protected:
    //Integer mu = 1, nu = 1; // Strides
    //Integer i0 = 0, j0 = 0; // Start indices
    //bool same_size = false; // Preserve original size (padding)

public:
    Integer nb_kers;
    Integer ker_size;

    Convolution(Integer nb_kers, Integer ker_size) : Layer(_Convolution), nb_kers(nb_kers), ker_size(ker_size){}

    Convolution* clone() const override { return new Convolution(*this); }

    // to be used when added to the network (it needs previous layer info to set the W matrix)
    void set_dims(Integer prev_channels, Integer prev_n);

    void forwardprop() override;
    void backprop() override;
    ~Convolution() = default;
};

class Pool : public Layer {
public:
    Integer p;
    Integer stride;
    TypePool typeP;

    Pool* clone() const override { return new Pool(*this); }
    Pool (TypePool t, Integer p, Integer stride) : Layer(_Pool), typeP(t), p(p), stride(stride) {}

    // to be used when added to the network (it needs previous layer info to set the dimensiosn of X)
    void set_dims(Integer prev_n, Integer prev_c);
    
    void forwardprop() override;
    void backprop() override; 
    
    // to be used when computing GradPrevX
    Reel get_dX_dPrevX_Conv() { return 1.0 / (p * p); }
    ~Pool() = default;
    
};

class Activation : public Layer {
public:
    TypeActivation typeA;
    FR_p fun_activation;
    FR_p dfun_activation;

    // constructor
    Activation(TypeActivation t) : Layer(_Activation), typeA(t) {
        fun_activation = (t == _relu) ? utils::relu : (t == _tanh) ? utils::hyper_tan : (t == _abs_tanh) ? utils::abs_hyper_tan : utils::sigmoid;
        dfun_activation = (t == _relu) ? utils::d_relu : (t == _tanh) ? utils::d_hyper_tan : (t == _abs_tanh) ? utils::d_abs_hyper_tan : utils::d_sigmoid;
    }
    Activation* clone() const override { return new Activation(*this); }

    // needs the previous layer info to set the dimensions of X
    void set_dims(Integer prev_d1, Integer prev_d2, Integer prev_d3, TypeLayer prev_type){
        dims[0] = prev_d1;
        dims[1] = prev_d2;
        dims[2] = prev_d3;
        if (prev_type == _Dense){ // rank 1
            X = RTensor(prev_d1);
            GradX = RTensor(prev_d1);
        }
        if (prev_type == _Convolution || prev_type == _Pool){ // rank 3
            X = RTensor(prev_d1, prev_d2, prev_d3);
            GradX = RTensor(prev_d1, prev_d2, prev_d3);
        }
    }

    void forwardprop() override;
    void backprop() override;
    RTensor get_dX_dPrevX_Dense() {
        Layer* prev = this->prevL(); // Get previous layer
        RTensor res = RTensor(X.size(), X.size()); // init with 2 args = rank set to 2 = matrix
        for (Integer i = 0; i < X.size(); ++i) { // diagonal matrix
            res(i, i) = dfun_activation(prev->X[i]);
        }
        return res;
    }
    
    Reel get_dX_dPrevX_Conv(Integer i, Integer j, Integer k, Integer i_p, Integer j_p, Integer c){
        if(X.rank() != 3){
            std::cerr << "Activation::get_dX_dPrevX_Conv: X is not a 3D Tensor" << std::endl;
        }
        if (i == i_p && j == j_p && k == c){
            return dfun_activation(X(i, j, k));
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
        X = RTensor(n); // here, we know for sure to init X as rank 1
        GradX = RTensor(n);
    }
    void set_dims(Integer prev_size);
    Dense* clone() const override;
    void forwardprop() override;
    void backprop() override;
    RTensor get_dX_dPrevX_Dense() { return W.transpose(); } // W should be of rank 2 (matrix)
    //void print(std::ostream& out) const override {}
    ~Dense() = default;
};

class Loss : public Layer{
protected:
    TypeLoss typeP;
    FV2_p fun_Loss;
    FV2V_p dfun_Loss;
    RTensor vref;
public:
    // constructor
    Loss(TypeLoss t1) : Layer(_Loss, 1, 1, 1), typeP(t1) { // X is just 1 number
        setFunPtr();
        X = RTensor(1);
        GradX = RTensor(1);
    }
    RTensor get_vref() const { return vref; }
    void set_vref(const RTensor& v) { vref = v; }
    FV2_p get_fun_Loss() const { return fun_Loss; }
    FV2V_p get_dfun_Loss() const { return dfun_Loss; }
    void setFunPtr();
    Loss* clone() const override { return new Loss(*this); }
    void forwardprop() override;
    void backprop() override {} // no backprop because no parameters
    //virtual void print(std::ostream& out) const;
    ~Loss() = default;
};


class Flatten : public Layer {
    public:
        // constructor
        Flatten() : Layer(_Flatten) {};
        void set_dims(Integer prev_d1, Integer prev_d2, Integer prev_d3){
            dims[0] = prev_d1 * prev_d2 * prev_d3;
            dims[1] = 1;
            dims[2] = 1;
            X = RTensor(dims[0]);
            GradX = RTensor(dims[0]);
        }
        Reel get_dX_dPrevX_Conv(Integer p, Integer i, Integer j, Integer k, Integer prev_n){
            if (p == i*prev_n*prev_n + j*prev_n + k){
                return 1;
            }
            return 0;
        }
        Flatten* clone() const override { return new Flatten(*this); }
        void forwardprop() override;
        void backprop() override;
        ~Flatten() = default;
    };

#endif // LAYERS_H
