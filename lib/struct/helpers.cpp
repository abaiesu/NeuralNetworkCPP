/**
 * @brief Computes the learning rate based on the step type.
 *
 * @param tp Step type.
 * @param rho Decay rate.
 * @param alpha Base learning rate.
 * @param k Current iteration.
 * @return Computed learning rate.
 */
 Reel computeLearningRate(int tp, Reel rho, Reel alpha, Integer k) {
    switch (tp) {
        case _fixed:       return alpha;
        case _linear:      return alpha / static_cast<Reel>(k);
        case _quadratic:   return alpha / (static_cast<Reel>(k) * static_cast<Reel>(k));
        case _exponential: return alpha * std::exp(-rho * static_cast<Reel>(k));
        default:           return alpha;
    }
}

/**
 * @brief Applies the convolution kernel with padding at a given output location.
 *
 * @param input Flattened input vector.
 * @param in_rows Number of rows in the input.
 * @param in_cols Number of columns in the input.
 * @param out_i Output row index.
 * @param out_j Output column index.
 * @param kernel Convolution kernel.
 * @return Convolved value.
 */
 Reel kerPad(const RVector &input, Integer in_rows, Integer in_cols,
                            Integer out_i, Integer out_j,
                            const RMatrix &kernel) {
    Integer k_rows = kernel.size();
    Integer k_cols = (k_rows > 0 ? kernel[0].size() : 0);
    Integer pad_h = k_rows / 2;
    Integer pad_w = k_cols / 2;
    Reel sum = 0;
    for (Integer u = 0; u < k_rows; u++) {
        for (Integer v = 0; v < k_cols; v++) {
            Integer in_i = out_i + u - pad_h;
            Integer in_j = out_j + v - pad_w;
            if (in_i < 0 || in_i >= in_rows || in_j < 0 || in_j >= in_cols)
                continue;
            sum += input[in_i * in_cols + in_j] * kernel[u][v];
        }
    }
    return sum;
}

/**
 * @brief Applies the convolution kernel at a given output location no padding.
 *
 * @param input Flattened input vector.
 * @param in_rows Number of rows in the input.
 * @param in_cols Number of columns in the input.
 * @param out_i Output row index.
 * @param out_j Output column index.
 * @param kernel Convolution kernel.
 * @param start_i Starting row index in the input.
 * @param start_j Starting column index in the input.
 * @param mu Vertical stride.
 * @param nu Horizontal stride.
 * @return Convolved value.
 */
 Reel kerNoPad(const RVector &input, Integer in_rows, Integer in_cols,
                             Integer out_i, Integer out_j,
                             const RMatrix &kernel,
                             Integer start_i, Integer start_j,
                             Integer mu, Integer nu) {
    Integer k_rows = kernel.size();
    Integer k_cols = (k_rows > 0 ? kernel[0].size() : 0);
    Reel sum = 0;
    Integer input_i = start_i + out_i * mu;
    Integer input_j = start_j + out_j * nu;
    for (Integer u = 0; u < k_rows; u++) {
        for (Integer v = 0; v < k_cols; v++) {
            Integer cur_i = input_i + u;
            Integer cur_j = input_j + v;
            if (cur_i < in_rows && cur_j < in_cols)
                sum += input[cur_i * in_cols + cur_j] * kernel[u][v];
        }
    }
    return sum;
}

/**
 * @brief Accumulates gradients for padded backpropagation.
 *
 * @param input Flattened input vector.
 * @param in_rows Number of rows in the input.
 * @param in_cols Number of columns in the input.
 * @param out_i Output row index.
 * @param out_j Output column index.
 * @param kernel Convolution kernel.
 * @param gradKernel Matrix to accumulate kernel gradients.
 * @param gradInput Vector to accumulate input gradients.
 * @param grad_val Gradient value from the output.
 */
void accumulateGradPad(const RVector &input, Integer in_rows, Integer in_cols,
                               Integer out_i, Integer out_j,
                               const RMatrix &kernel,
                               RMatrix &gradKernel, RVector &gradInput,
                               Reel grad_val) {
    Integer k_rows = kernel.size();
    Integer k_cols = (k_rows > 0 ? kernel[0].size() : 0);
    Integer pad_h = k_rows / 2;
    Integer pad_w = k_cols / 2;
    for (Integer u = 0; u < k_rows; u++) {
        for (Integer v = 0; v < k_cols; v++) {
            Integer in_i = out_i + u - pad_h;
            Integer in_j = out_j + v - pad_w;
            if (in_i < 0 || in_i >= in_rows || in_j < 0 || in_j >= in_cols)
                continue;
            gradKernel(u, v) += input[in_i * in_cols + in_j] * grad_val;
            gradInput[in_i * in_cols + in_j] += kernel[u][v] * grad_val;
        }
    }
}

/**
 * @brief Accumulates gradients for non-padded backpropagation.
 *
 * @param input Flattened input vector.
 * @param in_rows Number of rows in the input.
 * @param in_cols Number of columns in the input.
 * @param out_i Output row index.
 * @param out_j Output column index.
 * @param kernel Convolution kernel.
 * @param gradKernel Matrix to accumulate kernel gradients.
 * @param gradInput Vector to accumulate input gradients.
 * @param grad_val Gradient value from the output.
 * @param start_i Starting row index in the input.
 * @param start_j Starting column index in the input.
 * @param mu Vertical stride.
 * @param nu Horizontal stride.
 */
void accumulateGradNoPad(const RVector &input, Integer in_rows, Integer in_cols,
                                Integer out_i, Integer out_j,
                                const RMatrix &kernel,
                                RMatrix &gradKernel, RVector &gradInput,
                                Reel grad_val,
                                Integer start_i, Integer start_j,
                                Integer mu, Integer nu) {
    Integer k_rows = kernel.size();
    Integer k_cols = (k_rows > 0 ? kernel[0].size() : 0);
    Integer input_i = start_i + out_i * mu;
    Integer input_j = start_j + out_j * nu;
    for (Integer u = 0; u < k_rows; u++) {
        for (Integer v = 0; v < k_cols; v++) {
            Integer cur_i = input_i + u;
            Integer cur_j = input_j + v;
            if (cur_i < in_rows && cur_j < in_cols) {
                gradKernel(u, v) += input[cur_i * in_cols + cur_j] * grad_val;
                gradInput[cur_i * in_cols + cur_j] += kernel[u][v] * grad_val;
            }
        }
    }
}