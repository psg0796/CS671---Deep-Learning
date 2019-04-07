import tensorflow as tf
import numpy as np
import sklearn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import time
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.applications.resnet50 import ResNet50

filepath = 'model/model'

model = tf.keras.models.load_model(
    filepath,
    compile=True
)

(x_train, y_train) = (np.load('input/x_train.npy'), np.load('input/y_train.npy'))
(x_test, y_test) = (np.load('input/x_test.npy'), np.load('input/y_test.npy'))

init = tf.global_variables_initializer()



print (model.summary())


with tf.Session() as s:
  
	s.run(init)
	x = x_test[5]
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	print(x.shape)

	preds = model.predict(x)
	# print(preds.shape)
	# print ("Predicted: ", decode_predictions(preds, top=3)[0])

	#985 is the class index for class 'Daisy' in Imagenet dataset on which my model is pre-trained
	flower_output = model.output[:, 30]
	last_conv_layer = model.get_layer('conv2d_3')

	grads = K.gradients(flower_output, last_conv_layer.output)[0]
	pooled_grads = K.mean(grads, axis=(0, 1, 2))
	iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
	pooled_grads_value, conv_layer_output_value = iterate([x])

	#2048 is the number of filters/channels in 'mixed10' layer
	for i in range(16):
		conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

	heatmap = np.mean(conv_layer_output_value, axis=-1)
	heatmap = np.maximum(heatmap, 0)
	heatmap /= np.max(heatmap)
	plt.imshow(heatmap)
	# print(heatmap.shape)
	# plt.savefig(heatmap)

	#Using cv2 to superimpose the heatmap on original image to clearly illustrate activated portion of image
	img = x
	# heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
	# heatmap = np.uint8(255 * heatmap)
	# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
	# superimposed_img = heatmap * 0.4 + img
	# cv2.imwrite('image_name.jpg', superimposed_img)