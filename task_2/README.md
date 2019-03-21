This file contains the solution for task 3 given in the problem statement.

## Problem Statement
CS671_Assignment_1__2019_.pdf part 3.

## Code File
task3.py

## Available Script
### `task3.py`
Running this script will load the mnist dataset, runs the model (training, validating and testing) and output the result for test accuracy.

## Requirements
* numpy
* keras
* matplotlib

## Function Definitions
### `load_dataset()`
This function loads the mnist dataset using keras, normalize x values and flatten them.
> return values
>> * X_train : training input features
>> * y_train : training output classes
>> * X_val : validation input features
>> * y_val : validation output classes
>> * X_test : test input features
>> * y_test : test output classes

### `Sigmoidal()`
This function takes an integer as  an input and returns the sigmoidal value of that input.
### `MSE()`
This function gives the squared difference between two inputs.
### `encode()`
This function is used to produce the one hot encoding of certain input class which is a integer.
### `main()`
The driver function for the task3. It loads the dataset, trains the model, validate it and finally test on the testing data.

## Class Definitions
### `Layer`
This class contains variables and functions related to the layer. `find_output()` function is used to calculate and store the output of a certain layer. `adjust_weights()` function is used to modify the weights of a layer while back propagating.
### `Model`
This class contains variables and functions used for training, testing and back-propagating through a model.