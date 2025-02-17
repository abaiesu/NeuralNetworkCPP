#ifndef BASIC_DATA_H
#define BASIC_DATA_H

#include <vector>
#include <iostream>

typedef float Reel; // Alias for real numbers
typedef size_t Integer; // Alias for positive integers


template <typename T>
class Vector : public std::vector<T> {
public:
    // Constructors
    Vector();
    Vector(size_t size, T defaultValue = T());
    
    // Overloaded operators
    Vector<T>& operator+=(const Vector<T>& other);
    Vector<T>& operator-=(const Vector<T>& other);
    Vector<T>& operator*=(T scalar);
    Vector<T>& operator/=(T scalar);
    
    // Dot product operator
    T operator|(const Vector<T>& other) const;
    
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
private:
    size_t rows, cols;
public:
    // Constructors
    Matrix(size_t rows, size_t cols, T defaultValue = T());
    
    // Access element at (i, j)
    T& operator()(size_t i, size_t j);
    const T& operator()(size_t i, size_t j) const;
    
    // Overloaded operators
    Matrix<T>& operator+=(const Matrix<T>& other);
    Matrix<T>& operator-=(const Matrix<T>& other);
    Matrix<T>& operator*=(T scalar);
    Matrix<T>& operator/=(T scalar);
    
    // Matrix-vector and vector-matrix multiplication
    Vector<T> operator*(const Vector<T>& vec) const;
    Matrix<T> operator*(const Matrix<T>& other) const;

    Matrix<T> transpose() const;
    
    friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat) {
        for (size_t i = 0; i < mat.rows; ++i) {
            for (size_t j = 0; j < mat.cols; ++j) {
                os << mat(i, j) << " ";
            }
            os << std::endl;
        }
        return os;
    }
};

typedef Matrix<Reel> RMatrix; // Alias for real-valued matrices

RMatrix outerprod(const RVector& a, const RVector& b) {
    RMatrix result(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            result(i, j) = a[i] * b[j];
        }
    }
    return result;
}

#endif // BASIC_DATA_H