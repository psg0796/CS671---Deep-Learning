import tensorflow as tf
import numpy as np
import sklearn

filepath = '../task2/model6/model'
sess = tf.Session()
activation = []
activation_head = [[],[],[],[]]
inputPath = '../task2/inputData/'
(x_train, y_train_length, y_train_width, y_train_color, y_train_angle) = (np.load(inputPath + 'x_train.npy'), np.load(inputPath + 'y_train_length.npy'), np.load(inputPath + 'y_train_width.npy'), np.load(inputPath + 'y_train_color.npy'), np.load(inputPath + 'y_train_angle.npy'))
(x_test, y_test_length, y_test_width, y_test_color, y_test_angle) = (np.load(inputPath + 'x_test.npy'), np.load(inputPath + 'y_test_length.npy'), np.load(inputPath + 'y_test_width.npy'), np.load(inputPath + 'y_test_color.npy'), np.load(inputPath + 'y_test_angle.npy'))

with sess.as_default():
    model = tf.keras.models.load_model(
        filepath,
        compile=True
    )
    
    inputData = tf.cast(x_train[:1],tf.float32)
    for i in range(4):
        layer = model.get_layer(index = i)
        layerActivation = layer(inputData)
        activation.append(layerActivation.eval())
        inputData = layerActivation
        
    input_data_head = [inputData, inputData, inputData, inputData]
    for i in range(0,4):
        for j in range(0,4):
            layer = model.get_layer(index=4*i + j + 4)
            layerActivation = layer(input_data_head[j])
            input_data_head[j] = layerActivation
            activation_head[j].append(layerActivation.eval())