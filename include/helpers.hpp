#ifndef HELPERS_H
#define HELPERS_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include "basic_data.hpp"
#include "decl.hpp"


namespace utils {

    // ---------------------------- ACTIVATION FUNCTIONS ----------------------------
    Reel relu(Reel x);
    Reel d_relu(Reel x);
    Reel hyper_tan(Reel x);
    Reel d_hyper_tan(Reel x);
    Reel abs_hyper_tan(Reel x);
    Reel d_abs_hyper_tan(Reel x);
    Reel sigmoid(Reel x);
    Reel d_sigmoid(Reel x);

    // ---------------------------- MISC ----------------------------

    Reel computeLearningRate(TypeStep tp, Reel rho, Reel alpha, Integer k);
    void shuffle_data(std::vector<RTensor>& Es, std::vector<RTensor>& Ss);
    RTensor outerProduct(const RTensor& a, const RTensor& b);
    RTensor convolve(const RTensor& ker, const RTensor& X);

    // ---------------------------- LOSS FUNCTIONS ----------------------------

    Reel moindre_carre(const RTensor& y, const RTensor& y_pred);
    RTensor d_moindre_carre(const RTensor& y, const RTensor& y_pred);
    Reel moindre_abs(const RTensor& y, const RTensor& y_pred);
    RTensor d_moindre_abs(const RTensor& y, const RTensor& y_pred);
    Reel entropie_croisee(const RTensor& y, const RTensor& y_pred);
    RTensor d_entropie_croisee(const RTensor& y, const RTensor& y_pred);
}

std::ostream& operator<<(std::ostream& os, TypeLayer type);

#endif // HELPERS_H
