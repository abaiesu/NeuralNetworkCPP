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


template <typename T>
class Tensor : public Vector<T> {
public:
    size_t dim1;  // e.g. height
    size_t dim2;  // e.g. width
    size_t dim3;  // e.g. depth
    size_t dim4;  // e.g. channels [ = 1 if using a 3D-only constructor ]

    // -------------------------
    // Constructors
    // -------------------------
    Tensor() : dim1(0), dim2(0), dim3(0), dim4(0) {}

    // 3D constructor: (d1 x d2 x d3), dim4 = 1
    Tensor(size_t d1, size_t d2, size_t d3, T defaultValue = T())
        : dim1(d1), dim2(d2), dim3(d3), dim4(1)
    {
        this->resize(d1 * d2 * d3, defaultValue);
    }

    // 4D constructor: (d1 x d2 x d3 x d4)
    Tensor(size_t d1, size_t d2, size_t d3, size_t d4, T defaultValue = T())
        : dim1(d1), dim2(d2), dim3(d3), dim4(d4)
    {
        this->resize(d1 * d2 * d3 * d4, defaultValue);
    }

    Tensor(const RVector &vec, size_t d1, size_t d2, size_t d3)
        : dim1(d1), dim2(d2), dim3(d3)
    {
        if (vec.size() != d1 * d2 * d3) {
            throw std::invalid_argument("Tensor size mismatch");
        }
        this->resize(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            (*this)[i] = vec[i];
        }
    }

    // -------------------------
    // Indexing operator
    // -------------------------
    T& operator()(size_t i, size_t j, size_t k, size_t c = 0) {
        // Optional: range-checking
        // if (i >= dim1 || j >= dim2 || k >= dim3 || c >= dim4) {...}
        return (*this)[((i * dim2 + j) * dim3 + k) * dim4 + c];
    }

    const T& operator()(size_t i, size_t j, size_t k, size_t c = 0) const {
        return (*this)[((i * dim2 + j) * dim3 + k) * dim4 + c];
    }

    // -------------------------
    // slice2D methods
    // -------------------------
    // 1) For a 3D tensor (dim4 == 1): fix k, get the (dim1 x dim2) plane
    //    i.e. X(i,j,k) for i=0..dim1-1, j=0..dim2-1
    RMatrix slice2D(size_t d3Index) const {
        if (dim4 != 1) {
            throw std::runtime_error("slice2D(d3Index) only valid if dim4==1 (3D tensor).");
        }
        if (d3Index >= dim3) {
            throw std::out_of_range("d3Index out of range");
        }

        RMatrix out(dim1, dim2);
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                out(i, j) = (*this)(i, j, d3Index);
            }
        }
        return out;
    }

    // 2) For a 4D tensor: fix (k, c) -> get the (dim1 x dim2) plane
    //    i.e. X(i,j,k,c) for i=0..dim1-1, j=0..dim2-1
    RMatrix slice2D(size_t d3Index, size_t d4Index) const {
        if (dim4 <= 1) {
            throw std::runtime_error("slice2D(d3Index, d4Index) only valid if dim4>1 (4D tensor).");
        }
        if (d3Index >= dim3 || d4Index >= dim4) {
            throw std::out_of_range("Indices out of range");
        }

        RMatrix out(dim1, dim2);
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                out(i, j) = (*this)(i, j, d3Index, d4Index);
            }
        }
        return out;
    }

    // -------------------------
    // setSlice2D methods
    // -------------------------
    // The counterparts to 'slice2D' that let you write data back.
    void setSlice2D(size_t d3Index, const RMatrix& mat2D) {
        // Only valid if dim4 == 1, i.e. a 3D tensor
        if (dim4 != 1) {
            throw std::runtime_error("setSlice2D(d3Index, mat2D) only valid if dim4==1 (3D).");
        }
        if (mat2D.rows != dim1 || mat2D.cols != dim2) {
            throw std::runtime_error("setSlice2D: dimension mismatch");
        }
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                (*this)(i, j, d3Index) = mat2D(i, j);
            }
        }
    }

    void setSlice2D(size_t d3Index, size_t d4Index, const RMatrix& mat2D) {
        // For a 4D tensor
        if (dim4 <= 1) {
            throw std::runtime_error("setSlice2D(d3Index, d4Index, mat2D) only valid if dim4>1 (4D).");
        }
        if (mat2D.rows != dim1 || mat2D.cols != dim2) {
            throw std::runtime_error("setSlice2D: dimension mismatch");
        }
        if (d3Index >= dim3 || d4Index >= dim4) {
            throw std::out_of_range("Indices out of range");
        }
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                (*this)(i, j, d3Index, d4Index) = mat2D(i, j);
            }
        }
    }

    // ------------------------------------------------
    // print_shape, operator<<, etc. remain unchanged...
    // ------------------------------------------------

    void print_shape(std::ostream& os = std::cout) const {
        os << "(" << dim1 << " x " << dim2 << " x " << dim3;
        if (dim4 > 1) os << " x " << dim4;
        os << ")";
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
        for (size_t i = 0; i < tensor.dim1; ++i) {
            for (size_t j = 0; j < tensor.dim2; ++j) {
                for (size_t k = 0; k < tensor.dim3; ++k) {
                    for (size_t c = 0; c < tensor.dim4; ++c) {
                        os << tensor(i, j, k, c) << " ";
                    }
                    os << "| ";
                }
                os << std::endl;
            }
            os << "------------------" << std::endl;
        }
        return os;
    }
};


// Alias for a real-valued Tensor
typedef Tensor<Reel> RTensor;


#endif // BASIC_DATA_H