#ifndef DECL_H
#define DECL_H

#include "basic_data.hpp"

enum TypeLayer { _nondefini, _Entry, _Convolution, _Reduction, _Activation, _Dense, _Loss };
enum TypeReduction { _maxReduction, _moyenneReduction };
enum TypeActivation { _activation_indefini, _relu, _tanh, _abs_tanh, _sigmoid};
enum TypeLoss { _moindre_carre, _moindre_abs, _huber, _entropie_croisee, _softMax};
enum TypeStep {_fixed, _linear, _quadratic, _exponential};

typedef Reel (*FR_p)(Reel); // Function pointer for activation functions
typedef Reel (*FV2_p)(const RVector&, const RVector&); // Function pointer for loss functions
typedef RVector (*FV2V_p)(const RVector&, const RVector&); // Function pointer for loss gradients

#endif // DECL.HPP