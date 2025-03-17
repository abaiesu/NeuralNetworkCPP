#ifndef DECL_H
#define DECL_H

#include "basic_data.hpp"

enum TypeLayer { _nondefini, _Entry, _Convolution, _Pool, _Activation, _Dense, _Loss, _Flatten};
enum TypePool { _maxPool, _meanPool };
enum TypeActivation { _activation_indefini, _relu, _tanh, _abs_tanh, _sigmoid};
enum TypeLoss { _moindre_carre, _moindre_abs, _huber, _entropie_croisee, _softMax};
enum TypeStep {_fixed, _linear, _quadratic, _exponential};

typedef Reel (*FR_p)(Reel); // Function pointer for activation functions
typedef Reel (*FV2_p)(const RTensor&, const RTensor&); // Function pointer for loss functions
typedef RTensor (*FV2V_p)(const RTensor&, const RTensor&); // Function pointer for loss gradients

#endif // DECL.HPP