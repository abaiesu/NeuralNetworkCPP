#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <iostream>
#include <vector>
#include "basic_data.hpp"
#include "layers.hpp"


class Network{
    protected :
        vector<Layer*> layers; // liste des pointeurs des couches du Network
        string name= ""; // nom générique du fichier de sortie
        RVector residuals; // vecteur des residus
        Integer batch_size = -1; // -1 = pas de mini-batch
    public:
        // constructor
        Network() = default;
        Network(const string& name, Integer b) : name(name), batch_size(b) {}
        const vector<Layer*>&  getLayers() const { return layers; }
        void add(Layer* layer) { 
            layers.push_back(layer); 
            layer->network = this;
        }

        void forwardprop(const RVector& E, const RVector& S=Vecteur()); 
        void backprop(); // mise a jour des gradients G, Gpar
        void majParametres(TypePas tp,Reel rho,Reel alpha,Entier k); // iter. de gradient stochastique
        vector<vector<RVector>> batch_split(const vector<RVector>& Es, const vector<RVector>& Ss);
        void train_batch(const vector<RVector>& Es, const vector<RVector>& Ss);
        void train(const vector<RVector>&Es, const vector<RVector>&Ss, TypePas tp,
                    Reel rho=0.01, Reel alpha=0.001); //train the network
        void test(const vector<RVector>&Es, const vector<RVector>&Ss); //test
        void print(ostream&out) const;
    };

#endif // NETWORK_H