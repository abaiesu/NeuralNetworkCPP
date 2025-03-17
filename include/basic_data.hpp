#ifndef BASIC_DATA_HPP
#define BASIC_DATA_HPP

#include <vector>
#include <iostream>
#include <stdexcept>

using Reel = float;
using Integer = size_t;

template <typename T>
class Tensor : public std::vector<T> {
private:
    Integer _dims[4] = {0, 0, 0, 0};
    Integer rank_ = 0;

public:
    // Constructors for 1D, 2D, 3D, 4D tensors
    Tensor() = default;
    Tensor(Integer d1) : rank_(1) {
        _dims[0] = d1;
        this->resize(d1, 0);
    }
    Tensor(Integer d1, Integer d2) : rank_(2) {
        _dims[0] = d1; _dims[1] = d2;
        this->resize(d1 * d2, 0);
    }
    Tensor(Integer d1, Integer d2, Integer d3) : rank_(3) {
        _dims[0] = d1; _dims[1] = d2; _dims[2] = d3;
        this->resize(d1 * d2 * d3, 0);
    }
    Tensor(Integer d1, Integer d2, Integer d3, Integer d4) : rank_(4) {
        _dims[0] = d1; _dims[1] = d2; _dims[2] = d3; _dims[3] = d4;
        this->resize(d1 * d2 * d3 * d4, 0);
    }

    // Rank getter
    int rank() const { return rank_; }
    int dims(Integer d) const { return _dims[d]; }      
    //Integer size(Integer d) const { return (d < rank_) ? _dims[d] : 0; }

    // Access operators (read only)
    const T& operator()(Integer i) const {
        if (rank_ != 1) throw std::runtime_error("Rank mismatch in Tensor::operator()(i)");
        if (i >= _dims[0]) throw std::runtime_error("Index out of bounds in Tensor::operator()(i)");
        return (*this)[i];
    }
    const T& operator()(Integer i, Integer j) const {
        if (rank_ != 2) throw std::runtime_error("Rank mismatch in Tensor::operator()(i,j)");
        if (i >= _dims[0] || j >= _dims[1]) throw std::runtime_error("Index out of bounds in Tensor::operator()(i,j)");
        return (*this)[i * _dims[1] + j];
    }
    const T& operator()(Integer i, Integer j, Integer k) const {
        if (rank_ != 3) throw std::runtime_error("Rank mismatch in Tensor::operator()(i,j,k)");
        if (i >= _dims[0] || j >= _dims[1] || k >= _dims[2]) throw std::runtime_error("Index out of bounds in Tensor::operator()(i,j,k)");
        return (*this)[(i * _dims[1] + j) * _dims[2] + k];
    }
    const T& operator()(Integer i, Integer j, Integer k, Integer c) const {
        if (rank_ != 4) throw std::runtime_error("Rank mismatch in Tensor::operator()(i,j,k,c)");
        if (i >= _dims[0] || j >= _dims[1] || k >= _dims[2] || c >= _dims[3]) throw std::runtime_error("Index out of bounds in Tensor::operator()(i,j,k,c)");
        return (*this)[((i * _dims[1] + j) * _dims[2] + k) * _dims[3] + c];
    }

    // Access operators (modifable)
    T& operator()(Integer i) {
        if (rank_ != 1) throw std::runtime_error("Rank mismatch in Tensor::operator()(i)");
        if (i >= _dims[0]) throw std::runtime_error("Index out of bounds in Tensor::operator()(i)");
        return (*this)[i];
    }
    T& operator()(Integer i, Integer j) {
        if (rank_ != 2) throw std::runtime_error("Rank mismatch in Tensor::operator()(i,j)");
        if (i >= _dims[0] || j >= _dims[1]) throw std::runtime_error("Index out of bounds in Tensor::operator()(i,j)");
        return (*this)[i * _dims[1] + j];
    }
    T& operator()(Integer i, Integer j, Integer k) {
        if (rank_ != 3) throw std::runtime_error("Rank mismatch in Tensor::operator()(i,j,k)");
        if (i >= _dims[0] || j >= _dims[1] || k >= _dims[2]) throw std::runtime_error("Index out of bounds in Tensor::operator()(i,j,k)");
        return (*this)[(i * _dims[1] + j) * _dims[2] + k];
    }
    T& operator()(Integer i, Integer j, Integer k, Integer c) {
        if (rank_ != 4) throw std::runtime_error("Rank mismatch in Tensor::operator()(i,j,k,c)");
        if (i >= _dims[0] || j >= _dims[1] || k >= _dims[2] || c >= _dims[3]) throw std::runtime_error("Index out of bounds in Tensor::operator()(i,j,k,c)");
        return (*this)[((i * _dims[1] + j) * _dims[2] + k) * _dims[3] + c];
    }

    // Scalar Multiplication
    Tensor<T>& operator*=(T scalar) {
        for (auto& val : *this) val *= scalar;
        return *this;
    }

    // Matrix Transpose (only for rank 2)
    Tensor<T> transpose() const {
        if (rank_ != 2) throw std::runtime_error("Transpose only applies to 2D tensors");
        Tensor<T> result(_dims[1], _dims[0]);
        for (Integer i = 0; i < _dims[0]; ++i) {
            for (Integer j = 0; j < _dims[1]; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    Tensor<T> slice_2d(Integer dim3_idx, Integer dim4_idx) const {
        if (rank_ != 4) {
            throw std::runtime_error("2D slicing with 2 depths is only applicable to 4D tensors.");
        }

        Tensor<T> result(_dims[0], _dims[1]);

        for (Integer i = 0; i < _dims[0]; ++i) {
            for (Integer j = 0; j < _dims[1]; ++j) {
                result(i, j) = (*this)(i, j, dim3_idx, dim4_idx);
            }
        }

        return result;
    }

    Tensor<T> slice_2d(Integer dim3_idx) const {
        if (rank_ != 3) {
            throw std::runtime_error("2D slicing with 1 depth is only applicable to 3D tensors.");
        }

        Tensor<T> result(_dims[0], _dims[1]);

        for (Integer i = 0; i < _dims[0]; ++i) {
            for (Integer j = 0; j < _dims[1]; ++j) {
                result(i, j) = (*this)(i, j, dim3_idx);
            }
        }
        
        return result;
    }
    
    void set_channel(Integer c, const Tensor<T>& slice) {
        if (rank_ != 3) {
            throw std::runtime_error("Channel assignment is only applicable to 3D tensors.");
        }

        if (slice.dims(0) != _dims[0] || slice.dims(1) != _dims[1]) {
            throw std::runtime_error("Channel assignment requires a tensor of the same dimensions.");
        }

        for (Integer i = 0; i < _dims[0]; ++i) {
            for (Integer j = 0; j < _dims[1]; ++j) {
                (*this)(i, j, c) = slice(i, j);
            }
        }
    }


    Tensor<T> operator+=(const Tensor<T>& other) {
        if (rank_ != other.rank_) {
            throw std::runtime_error("(+=) Tensor ranks do not match.");
        }

        if (this->size() != other.size()) {
            throw std::runtime_error("(+=) Tensor sizes do not match.");
        }

        for (Integer i = 0; i < this->size(); ++i) {
            (*this)[i] += other[i];
        }

        return *this;
    }

    Tensor<T> operator-=(const Tensor<T>& other) {
        if (rank_ != other.rank_) {
            throw std::runtime_error("(-=) Tensor ranks do not match.");
        }

        if (this->size() != other.size()) {
            throw std::runtime_error("(-=) Tensor sizes do not match.");
        }

        for (Integer i = 0; i < this->size(); ++i) {
            (*this)[i] -= other[i];
        }

        return *this;
    }


    // Print Tensor Shape
    void print_shape() const {
        std::cout << "(";
        for (Integer i = 0; i < rank_; ++i) {
            std::cout << _dims[i];
            if (i + 1 < rank_) std::cout << " x ";
        }
        std::cout << ")\n";
    }

    // Matrix-Vector Multiplication
    friend Tensor<T> operator*(const Tensor<T>& mat, const Tensor<T>& vec) {
        //std::cout << "Matrix rank: " << mat.rank_ << ", Vector rank: " << vec.rank_ << std::endl;
        if (mat.rank_ != 2 || vec.rank_ != 1)
            throw std::invalid_argument("Matrix-vector multiplication requires a 2D matrix and a 1D vector");
        Integer rows = mat._dims[0];
        Integer cols = mat._dims[1];
        if (cols != vec._dims[0])
            throw std::invalid_argument("Matrix columns must match vector size.");

        Tensor<T> result(rows);
        for (Integer i = 0; i < rows; ++i) {
            T sum = 0;
            for (Integer j = 0; j < cols; ++j) {
                sum += mat(i, j) * vec(j);
            }
            result(i) = sum;
        }
        return result;
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
        if (tensor.rank_ == 1) {
            os << "[";
            for (Integer i = 0; i < tensor._dims[0]; ++i) {
                os << tensor(i);
                if (i + 1 < tensor._dims[0]) os << ", ";
            }
            os << "]";
        } else if (tensor.rank_ == 2) {
            os << "[\n";
            for (Integer i = 0; i < tensor._dims[0]; ++i) {
                os << "  [";
                for (Integer j = 0; j < tensor._dims[1]; ++j) {
                    os << tensor(i, j);
                    if (j + 1 < tensor._dims[1]) os << ", ";
                }
                os << "]";
                if (i + 1 < tensor._dims[0]) os << ",\n";
            }
            os << "\n]";
        } else {
            os << "TODO";
        }
        return os;
    }
};

// Alias for floating-point tensors
using RTensor = Tensor<Reel>;

#endif // RTENSOR_HPP