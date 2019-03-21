import numpy as np
import math

import keras
import matplotlib.pyplot as plt

def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # normalize x
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.

    # we reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

    return X_train, y_train, X_val, y_val, X_test, y_test

def Sigmoidal(x):
    try:
        ans = 1.0/(1 + math.exp(-x))
    except OverflowError:
        if x < 0:
            ans = 0
        else:
            ans = 1
    return ans

def MSE(y1, y2):
    # deviding by 2 for better derivative
    return ((y1 - y2)**2)/2.0

class Layer:
    def __init__(self, layer_type = "inner", num_of_neurons = 16, x = np.transpose(np.matrix([np.zeros(16)])), activation_func = Sigmoidal, next_layer = None, pre_layer = None):
        mu = 0
        sigma = 1
        self.layer_type = layer_type
        self.next_layer = next_layer
        self.pre_layer = pre_layer
        # self.x = [1, x1, x2, x3 .....]
        # dimensions:   (n+1) X 1
        self.x = x # np.insert(x, 0, 1, 0)
        self.activation_func = activation_func
        # dimensions:   m X (n+1)
        self.W = np.random.normal(mu, sigma, (num_of_neurons, np.size(self.x, 0)))
        self.output = np.transpose(np.matrix([np.zeros(num_of_neurons)]))

    def find_output(self):
        # dimensions:   m X 1
        output = np.dot(self.W, self.x)
        for i in range(np.size(output, 0)):
            output[i][0] = self.activation_func(output[i][0])
        self.output = output
        return self.output
    
    def adjust_weights(self, alpha):
        # new_W = np.subtract(self.W, alpha * self.derivative[:1,])
        new_W = np.subtract(self.W, alpha * self.derivative)
        self.W = new_W
    
    def calculate_error(self, target = None, cost_function = MSE,):
        cost = 0
        output = self.output
        for i in range(np.size(target, 0)):
            x = np.asscalar(target[i][0])
            y = np.asscalar(output[i][0])
            cost += cost_function(x, y)
        return cost

    def find_max_output_neuron_index(self):
        return np.asscalar(np.argmax(self.output, 0))

    # params =  {back_derivative = np.matrix([]), target = np.matrix([]), W = np.matrix([])}
    def find_derivative(self, params):
        if self.layer_type == "outer":
            target = params['target']
            derivative_multiplier = self.output
            for i in range(np.size(derivative_multiplier, 0)):
                derivative_multiplier[i][0] = -((target[i][0] - derivative_multiplier[i][0]) * (1 - derivative_multiplier[i][0]))
            self.derivative = np.dot(derivative_multiplier, np.transpose(self.x))
            return self.derivative
        else:
            x1 = self.next_layer.x
            x2 = np.transpose(self.x)
            for i in range(np.size(x1, 0)):
                if x1[i][0] != 0:
                    x1[i][0] = (1-x1[i][0])/x1[i][0]
                else:
                    x1[i][0] = 1000000000
            derivative_multiplier = np.dot(np.dot(params['W'], x1), x2)
            self.derivative = np.dot(np.transpose(params['back_derivative']), derivative_multiplier)
            return self.derivative


class Model:
    def __init__(self, input_size, num_of_classes, num_of_layers = 3):
        self.alpha = 0.4
        self.input_layer = Layer("inner", 100, np.transpose(np.matrix([np.zeros(input_size)])))
        layer_iterator = self.input_layer
        layer_iterator.pre_layer = None
        for i in range(1, num_of_layers):
            layer_iterator.next_layer = Layer("inner", 10, layer_iterator.output)
            layer_iterator.next_layer.pre_layer = layer_iterator
            layer_iterator = layer_iterator.next_layer
        layer_iterator.next_layer = Layer("outer", num_of_classes, layer_iterator.output)
        layer_iterator.next_layer.pre_layer = layer_iterator
        self.last_layer = layer_iterator.next_layer
        self.last_layer.next_layer = None
    
    def train(self, input, target):
        layer_target = np.transpose(target)
        layer_input = np.transpose(input)
        layer_iterator = self.input_layer
        target_output_class = np.asscalar(np.argmax(layer_target, 0))
        while layer_iterator != None:
            layer_iterator.x = layer_input #np.insert(layer_input, 0, 1, 0)
            layer_input = layer_iterator.find_output()
            layer_iterator = layer_iterator.next_layer
        self.back_propagate(layer_target)
        if self.last_layer.find_max_output_neuron_index() != target_output_class:
            return 0
        else:
            return 1
            
    def back_propagate(self, target):
        layer_iterator = self.last_layer
        layer_iterator.find_derivative({
            "target": target
        })
        layer_iterator.adjust_weights(self.alpha)
        layer_iterator = layer_iterator.pre_layer
        while layer_iterator.pre_layer != None:
            layer_iterator.find_derivative({
                "back_derivative": layer_iterator.next_layer.derivative,
                "W": layer_iterator.next_layer.W
            })
            layer_iterator.adjust_weights(self.alpha)
            layer_iterator = layer_iterator.pre_layer

    def test(self, input):
        layer_input = np.transpose(input)
        layer_iterator = self.input_layer
        while layer_iterator != None:
            layer_iterator.x = layer_input #np.insert(layer_input, 0, 1, 0)
            layer_input = layer_iterator.find_output()
            layer_iterator = layer_iterator.next_layer
        output_class = self.last_layer.find_max_output_neuron_index()
        print(output_class)

def encode(x):
    vec = np.zeros(10)
    vec[x] = 1
    return vec

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten=True)
    input_size = np.size(X_train, 1)
    output_classes = 10
    model = Model(input_size, output_classes)
    num_of_training_data = np.size(X_train, 0)
    accuracy = []
    resultFile = open("results.txt", "a")
    resultFile.write("Training..................................\n")
    resultFile.close()
    for j in range(10):
        for i in range(num_of_training_data):
            input = X_train[i]
            target = encode(y_train[i])
            accuracy.append(model.train(np.matrix([input]), np.matrix([target])))
        accuracyFile = open("accuracy.txt", "a")
        accuracyFile.write("epoch " + str(j) + " acc " + str(np.average(accuracy[np.size(accuracy) - 10000:])) + "\n")
        accuracyFile.close()
        print(j)
    train_acc = np.average(accuracy[np.size(accuracy) - 10000:])
    accuracy = []
    resultFile = open("results.txt", "a")
    resultFile.write("Training accuracy \t" + str(train_acc * 100) + "\n")

    resultFile.write("Validatin..................................\n")
    resultFile.close()
    for i in range(np.size(X_val, 0)):
        input = X_val[i]
        target = y_val[i]
        model_output = model.test(np.matrix([input]))
        acc = 0
        if model_output == target:
            acc = 1
        accuracy.append(acc)
    val_acc = np.average(accuracy)
    accuracy = []
    resultFile = open("results.txt", "a")
    resultFile.write("Validation accuracy \t" + str(val_acc * 100) + "\n")

    resultFile.write("Testing..................................\n")    
    resultFile.close()
    for i in range(np.size(X_test, 0)):
        input = X_test[i]
        target = y_test[i]
        model_output = model.test(np.matrix([input]))
        acc = 0
        if model_output == target:
            acc = 1
        accuracy.append(acc)
    test_acc = np.average(accuracy)
    accuracy = []
    resultFile = open("results.txt", "a")
    resultFile.write("Test accuracy \t" + str(test_acc * 100) + "\n")
    resultFile.close()


if __name__ == "__main__":
    main()
