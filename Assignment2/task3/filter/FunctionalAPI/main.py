import tensorflow as tf
import numpy as np
import sklearn
from keras import backend as K
import cv2
import matplotlib.pyplot as plt
from time import time
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Dense, Dropout

filepath = '/home/psg/Workspace/IIT/Deep Learning/lab/CS671---Deep-Learning/Assignment2/task2/model6/model'

model = tf.keras.models.load_model(
    filepath,
    compile=True
)

print (model.summary())

a = np.load('/home/psg/Workspace/IIT/Deep Learning/lab/CS671---Deep-Learning/Assignment2/task2/inputData/x_test.npy')

test_image = a[5]

print(test_image.shape)

init = tf.global_variables_initializer()

def deprocess_image(x):
    
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index, size=150):
    
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    print(input_img_data.shape)
    step = 1.
    for i in range(80):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)

with tf.Session() as s:
  
    s.run(init)
    layer_name = 'conv2d_5'
    size = 28
    margin = 5

    fig = plt.figure(figsize=(20,20))
    plt.title("Filter")

    pos = 1
    for i in range(4):
        for j in range(8):
            filter_img = generate_pattern(layer_name, i + (j * 4), size=size)
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            fig.add_subplot(4,8,pos)
            pos += 1
            plt.imshow(filter_img)
            plt.grid(False)

plt.savefig("filters")