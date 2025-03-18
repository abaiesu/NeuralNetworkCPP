# NeuralNetworkCPP
Neural Network architecture in C++

Usage :

make
./output/main ex_number

1 : z = ax + bx

2 : z = 2 * x*x + 3 * y + 1

3 : z = sin (pi x) * cos (pi y) 

4 : CNN for z = 0.5 * avg(channel 1) + 2.0 * avg(channel 2) - 1.5 * avg(channel 3)

5 : CIFAR (to test that the code doesn't break), way too small for actual learning

The file test shows the results of the CNN architecture chosen for the CIFAR data set in Python. On the training data, 90% accuracy. On unseen data, around 50% (still better than a random guess).