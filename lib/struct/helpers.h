Reel computeLearningRate(int tp, Reel rho, Reel alpha, Integer k);

Reel kerPad(const RVector &input, Integer in_rows, Integer in_cols,
                            Integer out_i, Integer out_j,
                            const RMatrix &kernel);

Reel kerNoPad(const RVector &input, Integer in_rows, Integer in_cols,
                             Integer out_i, Integer out_j,
                             const RMatrix &kernel,
                             Integer start_i, Integer start_j,
                             Integer mu, Integer nu);

void accumulateGradPad(const RVector &input, Integer in_rows, Integer in_cols,
                               Integer out_i, Integer out_j,
                               const RMatrix &kernel,
                               RMatrix &gradKernel, RVector &gradInput,
                               Reel grad_val);

void accumulateGradNoPad(const RVector &input, Integer in_rows, Integer in_cols,
                                Integer out_i, Integer out_j,
                                const RMatrix &kernel,
                                RMatrix &gradKernel, RVector &gradInput,
                                Reel grad_val,
                                Integer start_i, Integer start_j,
                                Integer mu, Integer nu);
