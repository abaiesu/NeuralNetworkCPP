#ifndef BASIC_DATA_HPP
#define BASIC_DATA_HPP

#include <vector>
#include <iostream>

typedef float Reel; // Alias for real numbers
typedef size_t Integer; // Alias for positive integers


template <typename T>
class Vector : public std::vector<T> {
public:
    // Constructors
    Vector() = default;
    Vector(size_t len, T defaultValue = T()) {
        this->resize(len, defaultValue);
    }
    Vector(const std::vector<T>& vec) {
        this->resize(vec.size());
        for (size_t i = 0; i < this->size(); ++i) {
            (*this)[i] = vec[i];
        }
    }
    
    // Overloaded operators
    Vector<T>& operator+=(const Vector<T>& other){
        // check size compatibility
        if (this->size() != other.size()) {
            std::cerr << "Error: Vector sizes do not match! "
                  << "Current size: " << this->size()
                  << ", Other size: " << other.size() << std::endl;
            throw std::invalid_argument("Vector sizes do not match");
        }
        for (size_t i = 0; i < this->size(); ++i) {
            (*this)[i] += other[i];
        }
        return *this;
    };
    Vector<T>& operator-=(const Vector<T>& other){
        // check size compatibility
        if (this->size() != other.size()) {
            std::cerr << "Error: Vector sizes do not match! "
                  << "Current size: " << this->size()
                  << ", Other size: " << other.size() << std::endl;
            throw std::invalid_argument("Vector sizes do not match");
        }
        for (size_t i = 0; i < this->size(); ++i) {
            (*this)[i] -= other[i];
        }
        return *this;
    };
    Vector<T>& operator*=(T scalar){
        for (size_t i = 0; i < this->size(); ++i) {
            (*this)[i] *= scalar;
        }
        return *this;
    };
    Vector<T>& operator/=(T scalar){
        for (size_t i = 0; i < this->size(); ++i) {
            (*this)[i] /= scalar;
        }
        return *this;
    };
    
    // Dot product operator
    T operator|(const Vector<T>& other) const {
        // check size compatibility
        if (this->size() != other.size()) {
            std::cerr << "Error: Vector sizes do not match! "
                  << "Current size: " << this->size()
                  << ", Other size: " << other.size() << std::endl;
            throw std::invalid_argument("Vector sizes do not match");
        }
        T result = 0;
        for (size_t i = 0; i < this->size(); ++i) {
            result += (*this)[i] * other[i];
        }
        return result;
    }

    // Element-wise multiplication
    Vector<T> operator*(const Vector<T>& other) const {
        // check size compatibility
        if (this->size() != other.size()) {
            std::cerr << "Error: Vector sizes do not match! "
                  << "Current size: " << this->size()
                  << ", Other size: " << other.size() << std::endl;
            throw std::invalid_argument("Vector sizes do not match");
        }
        Vector<T> result(this->size());
        for (size_t i = 0; i < this->size(); ++i) {
            result[i] = (*this)[i] * other[i];
        }
        return result;
    }
    
    // Output stream overload
    friend std::ostream& operator<<(std::ostream& os, const Vector<T>& vec) {
        for (const auto& val : vec) {
            os << val << " ";
        }
        return os;
    }
};

typedef Vector<Reel> RVector; // Alias for real-valued vectors


template <typename T>
class Matrix : public Vector<T> {
public:
    size_t rows, cols;

    // Constructors
    Matrix() : rows(0), cols(0) {}
    Matrix(size_t rows, size_t cols, T defaultValue = T()) : rows(rows), cols(cols) {
        this->resize(rows * cols, defaultValue);
    }
    Matrix(RVector vec) : rows(1), cols(vec.size()) {
        this->resize(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            (*this)[i] = vec[i];
        }
    }
    
    // Access element at (i, j)
    T& operator()(size_t i, size_t j){ //modifiable 
        return (*this)[i * cols + j];
    }
    const T& operator()(size_t i, size_t j) const { // read only
        return (*this)[i * cols + j];
    }
    
    // Matrix-vector multiplication
    Vector<T> operator*(const Vector<T>& vec) const{
        // check size compatibility
        if (cols != vec.size()) {
            std::cerr << "Error: Sizes do not match!\n ";
            std::cerr << "Matrix: ";
            print_size(std::cerr);
            std::cerr << "Vector: " << vec.size() << std::endl;
            throw std::invalid_argument("Matrix and vector sizes do not match");
        }
        Vector<T> result(rows);
        for (size_t i = 0; i < rows; ++i) {
            result[i] = 0;
            for (size_t j = 0; j < cols; ++j) {
                result[i] += (*this)(i, j) * vec[j];
            }
        }
        return result;
    }

    Matrix<T> operator*(const Matrix<T>& other) const{
        // check size compatibility
        if (cols != other.rows) {
            std::cerr << "Error: Matrices cannot be multiplied! Dimensions do not match.\n";
            std::cerr << "First matrix: ";
            print_size(std::cerr);
            std::cerr << "Second matrix: ";
            other.print_size(std::cerr);
            throw std::invalid_argument("Matrix sizes do not match");
        }
        Matrix<T> result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                result(i, j) = 0;
                for (size_t k = 0; k < cols; ++k) {
                    result(i, j) += (*this)(i, k) * other(k, j);
                }
            }
        }
        return result;
    }

    Matrix<T> transpose() const{
        Matrix<T> result(cols, rows);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }
    
    friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat) {
        for (size_t i = 0; i < mat.rows; ++i) {
            for (size_t j = 0; j < mat.cols; ++j) {
                os << mat(i, j) << " ";
            }
            os << std::endl;
        }
        return os;
    }

    void print_size(std::ostream& os = std::cout) const {
        os << "(" << rows << " x " << cols << ")" << std::endl;
    }
    
};

typedef Matrix<Reel> RMatrix; // Alias for real-valued matrices

#endif // BASIC_DATA_H