#ifndef DATA_STRUC_H
#define DATA_STRUC_H

#include <vector>
#include <iostream>

typedef float Reel;     

// Template class for Vector
template <typename T>
class Vector : public std::vector<T> {
public:
    // Constructors
    Vector();
    Vector(size_t size, T defaultValue = T());
    
    // Overloaded operators (to be implemented)
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

typedef Vector<Reel> Vecteur; // Alias for real-valued vectors

// Template class for Matrix
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
    
    // Overloaded operators (to be implemented)
    Matrix<T>& operator+=(const Matrix<T>& other);
    Matrix<T>& operator-=(const Matrix<T>& other);
    Matrix<T>& operator*=(T scalar);
    Matrix<T>& operator/=(T scalar);
    
    // Matrix-vector and vector-matrix multiplication (to be implemented)
    Vector<T> operator*(const Vector<T>& vec) const;
    Matrix<T> operator*(const Matrix<T>& other) const;
    
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


#endif // DATA_STRUC_H