#include <iostream>
#include <vector>
#include <cstdlib>  
#include <ctime> 
#include <cmath>   
#include <fstream>
#include <string>
#include "layers.hpp"
#include "decl.hpp"

void readCifar10(const std::string &path, std::vector<RTensor> &images, std::vector<RTensor> &labels, int limit = 1000) {
    // CIFAR-10 format:
    //   Each record = 1 label byte + 3072 image bytes (32*32*3)
    //   We read 'limit' records from the file
    const int image_size  = 32 * 32 * 3;   // 3072
    const int record_size = 1 + image_size; // label + image

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file: " << path << std::endl;
        return;
    } else {
        std::cout << "Successfully opened file: " << path << std::endl;
    }

    // Allocate a buffer for one record
    char *buffer = new char[record_size];

    images.clear();
    labels.clear();
    images.reserve(limit);
    labels.reserve(limit);

    for (int i = 0; i < limit; i++) {
        // Read one record
        file.read(buffer, record_size);
        if (!file) {
            std::cerr << "Warning: reached end of file before reading all records." << std::endl;
            break;
        }

        // First byte = label
        unsigned char label_byte = static_cast<unsigned char>(buffer[0]);

        // Next 3072 bytes = 32 x 32 x 3 image
        // CIFAR order: first 1024 bytes = red, next 1024 = green, next 1024 = blue
        RTensor image(32, 32, 3);
        const int offset      = 1;        // first byte was label
        const int plane_size  = 32 * 32;  // 1024

        for (int ch = 0; ch < 3; ch++) {
            for (int px = 0; px < plane_size; px++) {
                int row = px / 32;
                int col = px % 32;
                // Normalize to [0,1]
                unsigned char pixel_val =
                    static_cast<unsigned char>(buffer[offset + ch * plane_size + px]);
                image(row, col, ch) = static_cast<Reel>(pixel_val) / 255.0;
            }
        }

        // Store the image
        images.push_back(image);

        // Create a one-hot vector for the label (10 classes)
        RTensor one_hot(10);
        for (int c = 0; c < 10; c++) {
            one_hot[c] = 0.0;
        }
        one_hot[label_byte] = 1.0;

        labels.push_back(one_hot);
    }

    delete[] buffer; // deallocate
    file.close();
}






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

        // The Entry layer now accepts a 3D input (8 x 8 x 3)
        Network network("ex4_3d_cnn", 32, 50);
        network.add(new Entry(8, 8, 3));
        network.add(new Convolution(4, 3));
        network.add(new Activation(_relu));
        network.add(new Pool(_meanPool, 2, 2));
        network.add(new Flatten());
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
            
    else if (example == 5) {
        std::cout << "Example 5 - CIFAR-10 classification" << std::endl;

        // We'll read from a single batch file (e.g., data_batch_1.bin)
        // Adjust path if needed
        std::string cifar_file = "src/data_batch_1.bin";

        // Prepare vectors to store images (32x32x3) and labels (10)
        std::vector<RTensor> data;
        std::vector<RTensor> labels;

        // Let's read 5000 images for training, for example
        int limit = 500;
        readCifar10(cifar_file, data, labels, limit);

        // Build a CNN for CIFAR-10 classification
        Network network("cifar10_classification", /*batchSize=*/32, /*epochs=*/10);

        // Input layer: 32x32x3
        network.add(new Entry(32, 32, 3));

        network.add(new Convolution(32, 3));
        network.add(new Activation(_relu));
        network.add(new Convolution(32, 3));
        network.add(new Activation(_relu));
        network.add(new Pool(_maxPool, 2, 2));
        network.add(new Convolution(64, 3));
        network.add(new Convolution(64, 3));
        network.add(new Flatten());
        network.add(new Dense(512));
        network.add(new Activation(_relu));
        network.add(new Dense(10)); 

        // Cross-entropy loss for classification
        network.add(new Loss(_cat_cross_entropy));

        network.print(std::cout);

        //std :: cout << "Shape input layer: " << data[0].dims(0) << " x " << data[0].dims(1) << " x " << data[0].dims(2) << std::endl;

        // Train
        network.train(data, labels, _fixed, /*learningRate=*/0.01, /*reg=*/0.001);

        // test
        // Prepare vectors to store images (32x32x3) and labels (10)
        std :: cout << "testing on the traing data" << std::endl;
        std::vector<RTensor> test_data;
        std::vector<RTensor> test_labels;

        limit = 10;
        readCifar10(cifar_file, test_data, test_labels, limit);

        //network.test(test_data, test_labels); // this will print the proba vector

        // Now, loop through each test example to obtain the predicted label
        int correct = 0;
        for (size_t i = 0; i < test_data.size(); i++) {
            // Forward propagate the test sample through the network
            network.forwardprop(test_data[i], test_labels[i]);
            
            // Assume the prediction is in the second-to-last layer (Dense layer before Loss)
            RTensor output = network.getLayers()[network.getLayers().size() - 2]->X;
            
            // Find the index of the maximum value in the output (predicted label)
            int pred_label = 0;
            double max_val = -1;
            for (int j = 1; j < output.size(); j++) {
                //std :: cout << "output[" << j << "] = " << output[j] << std::endl;
                //std :: cout << "max_val = " << max_val << std::endl;
                //Reel soft_maxed = std::exp(output[j]) / std::exp(output).sum();
                if (output[j] > max_val) {
                    max_val = output[j];
                    pred_label = j;
                }
            }
            
            // Convert the one-hot true label into a numeric label by finding the index of the max value
            RTensor true_onehot = test_labels[i];
            int true_label = 0;
            double max_true = true_onehot[0];
            for (int j = 1; j < true_onehot.size(); j++) {
                if (true_onehot[j] > max_true) {
                    max_true = true_onehot[j];
                    true_label = j;
                }
            }
            
            if (pred_label == true_label) {
                correct++;
            }
            std::cout << "Test sample " << i 
                    << ": Prediction = " << pred_label 
                    << ", True label = " << true_label << std::endl;
        }

    }




    else {
        std::cerr << "Invalid example number" << std::endl;
    }

    return 0;
}