#include <iostream>
#include <vector>
#include <cstdlib>  
#include <ctime> 
#include <cmath>   
#include <fstream>
#include <string>
#include "layers.hpp"
#include "decl.hpp"

double f_ex1(double x, double y, double a, double b) {
    return a * x + b * y;
}

double f_ex2(double x, double y) {
    return 2 * x*x + 3 * y + 1;
}

double f_ex3(double x, double y) {
    return std::sin(M_PI * x) * std::cos(M_PI * y);
}

void print(const std::string& message) {
    std::cout << message << std::endl;
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " example number (1, 2, 3)" << std::endl;
        return 1;
    }

    int example = std::atoi(argv[1]);

    if (example == 1) {
        
        print("Example 1");

        //RTensor test = RTensor(2, 2, 2);
        //std :: cout << "test size" << test.size() << std::endl;

        double a = 2;
        double b = 3;

        size_t n = 1000;

        // Generate random x and y values
        std::vector<RTensor> data(n, RTensor(2));  // Vector of pairs (x, y), each pair is a vector of size 2
        for (size_t i = 0; i < n; i++) {
            data[i][0] = static_cast<double>(std::rand() % 10) / 10.0;  // x ∈ [0, 1]
            data[i][1] = static_cast<double>(std::rand() % 100) / 100.0;  // y ∈ [0, 1]
        }

        // Compute the labels z using the f_ex1 function
        std::vector<RTensor> labels(n, RTensor(1));  // Vector of labels z, each label is a vector of size 1
        for (size_t i = 0; i < n; i++) {
            labels[i][0] = f_ex1(data[i][0], data[i][1], a, b);
        }

    
        Network network("f_ex1_regression", 32, 100);
        network.add(new Entry(2));              // Input layer with dimension 2
        network.add(new Dense(1));              // Dense layer with dimension 1
        network.add(new Loss(_moindre_carre));  // Loss layer with dimension 1 and loss type "moindre_carre"
        network.print(std::cout);               // Print the network
        network.train(data, labels, _fixed, 0.01, 0.001); // Train the network

        // last layer = loss layer
        // before last layer = dense layer with the predicted label
        Integer num_layers = network.getLayers().size();
        Dense* dense_layer = dynamic_cast<Dense*>(network.getLayers()[num_layers - 2]);
        double pred_a = dense_layer->W(0, 0);
        double pred_b = dense_layer->W(0, 1);

        std::cout << "Predicted a: " << pred_a << std::endl;
        std::cout << "Predicted b: " << pred_b << std::endl;

        // now test
        Integer num_test = 5;
        std::vector<RTensor> test_data(num_test, RTensor(2));  // Vector of pairs (x, y), each pair is a vector of size 2
        for (size_t i = 0; i < test_data.size(); i++) {
            test_data[i][0] = static_cast<double>(std::rand() % 10) ;  // Random x value
            test_data[i][1] = static_cast<double>(std::rand() % 100);  // Random y value
        }
        std::vector<RTensor> test_labels(num_test, RTensor(1));  // Vector of labels z, each label is a vector of size 1
        for (size_t i = 0; i < test_labels.size(); i++) {
            test_labels[i][0] = f_ex1(test_data[i][0], test_data[i][1], a, b);
        }

        network.test(test_data, test_labels);

    }else if (example == 2){
        
        print("Example 2");

        size_t n = 1000;

        // Generate random x and y values
        std::vector<RTensor> data(n, RTensor(2));  // Vector of pairs (x, y), each pair is a vector of size 2
        for (size_t i = 0; i < n; i++) {
            data[i][0] = static_cast<double>(std::rand() % 100) / 100;
            data[i][1] = static_cast<double>(std::rand() % 100) / 100;
        }

        // Compute the labels z using the f_ex2 function
        std::vector<RTensor> labels(n, RTensor(1));  
        for (size_t i = 0; i < n; i++) {
            labels[i][0] = f_ex2(data[i][0], data[i][1]);
        }

    
        Network network("ex2", 32, 100);
        network.add(new Entry(2));              // Input layer with dimension 2
        network.add(new Dense(16));              // Dense layer with dimension 1
        network.add(new Activation(_tanh));      // Activation layer with ReLU activation function
        network.add(new Dense(16));              // Dense layer with dimension 1
        network.add(new Activation(_relu));      // Activation layer with ReLU activation function
        network.add(new Dense(10));              // Dense layer with dimension 1
        network.add(new Activation(_tanh));      // Activation layer with ReLU activation function
        network.add(new Dense(1));              // Dense layer with dimension 1   
        network.add(new Loss(_moindre_carre));  // Loss layer with dimension 1 and loss type "moindre_carre"
        network.print(std::cout);               // Print the network
        network.train(data, labels, _fixed, 0.01, 0.001); // Train the network

        // now test
        Integer num_test = 5;
        std::vector<RTensor> test_data(num_test, RTensor(2));  // Vector of pairs (x, y), each pair is a vector of size 2
        for (size_t i = 0; i < test_data.size(); i++) {
            test_data[i][0] = static_cast<double>(std::rand() % 100) / 100;  // Random x value
            test_data[i][1] = static_cast<double>(std::rand() % 100) / 100;  // Random y value
        }


        std::vector<RTensor> test_labels(num_test, RTensor(1));  // Vector of labels z, each label is a vector of size 1
        for (size_t i = 0; i < test_labels.size(); i++) {
            test_labels[i][0] = f_ex2(test_data[i][0], test_data[i][1]);
        }

        network.test(test_data, test_labels);

    }else if (example == 3){
        
        print("Example 3");

        size_t n = 10000;

        // Generate random x and y values
        std::vector<RTensor> data(n, RTensor(2));  // Vector of pairs (x, y), each pair is a vector of size 2
        for (size_t i = 0; i < n; i++) {
            data[i][0] = static_cast<double>(std::rand() % 100) / 100;
            data[i][1] = static_cast<double>(std::rand() % 100) / 100;
        }

        // Compute the labels z using the f_ex2 function
        std::vector<RTensor> labels(n, RTensor(1));  
        for (size_t i = 0; i < n; i++) {
            labels[i][0] = f_ex2(data[i][0], data[i][1]);
        }

    
        Network network("ex3", 32, 100);
        
        network.add(new Entry(2));            
        network.add(new Dense(16));            
        network.add(new Activation(_tanh));     
        network.add(new Dense(32));              
        network.add(new Activation(_relu));     
        network.add(new Dense(64));             
        network.add(new Activation(_relu));    
        network.add(new Dense(16));
        network.add(new Activation(_relu));
        network.add(new Dense(1));              
        network.add(new Loss(_moindre_carre));  
        network.print(std::cout);              
        network.train(data, labels, _fixed, 0.01, 0.001); // Train the network

        // now test
        Integer num_test = 5;
        std::vector<RTensor> test_data(num_test, RTensor(2));  // Vector of pairs (x, y), each pair is a vector of size 2
        for (size_t i = 0; i < test_data.size(); i++) {
            test_data[i][0] = static_cast<double>(std::rand() % 100) / 100;  // Random x value
            test_data[i][1] = static_cast<double>(std::rand() % 100) / 100;  // Random y value
        }


        std::vector<RTensor> test_labels(num_test, RTensor(1));  // Vector of labels z, each label is a vector of size 1
        for (size_t i = 0; i < test_labels.size(); i++) {
            test_labels[i][0] = f_ex2(test_data[i][0], test_data[i][1]);
        }

        network.test(test_data, test_labels);

    }else if (example == 4) {

        print("Example 4 - 3D CNN regression");
    
        size_t n = 1000;
        // Create dataset: each input is a 3D tensor of shape 8 x 8 x 3
        std::vector<RTensor> data(n, RTensor(8, 8, 3));
        std::vector<RTensor> labels(n, RTensor(1));
    
        // Generate random 3D inputs and compute the corresponding label
        for (size_t i = 0; i < n; i++) {
            // Fill the tensor with random values between 0 and 1
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    for (int ch = 0; ch < 3; ch++) {
                        data[i](r, c, ch) = static_cast<double>(std::rand() % 101) / 100.0;
                    }
                }
            }
            // Compute channel-wise averages
            double sum0 = 0, sum1 = 0, sum2 = 0;
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    sum0 += data[i](r, c, 0);
                    sum1 += data[i](r, c, 1);
                    sum2 += data[i](r, c, 2);
                }
            }
            double avg0 = sum0 / (8 * 8);
            double avg1 = sum1 / (8 * 8);
            double avg2 = sum2 / (8 * 8);
            // Compute label as a weighted combination of the averages
            double output = 0.5 * avg0 + 2.0 * avg1 - 1.5 * avg2;
            labels[i](0) = output;
        }

        // Build a CNN network
        // The Entry layer now accepts a 3D input (8 x 8 x 3)
        Network network("ex4_3d_cnn", 32, 50);
        network.add(new Entry(8, 8, 3));
        // Add a convolution layer 4 kernels of size 3 x 3
        network.add(new Convolution(4, 3));
        network.add(new Activation(_relu));
        // Add a pooling layer (mean pooling with pool size 2 and stride 2)
        network.add(new Pool(_meanPool, 2, 2));
        // Flatten the 3D output into a vector
        network.add(new Flatten());
        // A Dense layer to map to 1 output
        network.add(new Dense(1));
        // Loss layer (using the mean squared error)
        network.add(new Loss(_moindre_carre));
    
        network.print(std::cout);
        
        network.train(data, labels, _fixed, 0.01, 0.001);
    
        // Test the network with new random data
        Integer num_test = 5;
        std::vector<RTensor> test_data(num_test, RTensor(8, 8, 3));
        std::vector<RTensor> test_labels(num_test, RTensor(1));
        for (size_t i = 0; i < num_test; i++) {
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    for (int ch = 0; ch < 3; ch++) {
                        test_data[i](r, c, ch) = static_cast<double>(std::rand() % 101) / 100.0;
                    }
                }
            }
            double sum0 = 0, sum1 = 0, sum2 = 0;
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    sum0 += test_data[i](r, c, 0);
                    sum1 += test_data[i](r, c, 1);
                    sum2 += test_data[i](r, c, 2);
                }
            }
            double avg0 = sum0 / (8 * 8);
            double avg1 = sum1 / (8 * 8);
            double avg2 = sum2 / (8 * 8);
            double output = 0.5 * avg0 + 2.0 * avg1 - 1.5 * avg2;
            test_labels[i][0] = output;
        }
    
        network.test(test_data, test_labels);
    }
            
    else {
        std::cerr << "Invalid example number" << std::endl;
    }
    return 0;
}
    
