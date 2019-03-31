import tensorflow as tf
import cv2
import numpy as np

imgDir_train = '../input_train/'
imgDir_test = '../input_test/'
imgFmt = 'jpg'

# Format of image = angle_length_width_color_variation

angle = 12
length = 2
width = 2
color = 2
variation_train = 600
variation_test = 400

x_train = []
y_length_train = []
y_color_train = []
y_angle_train = []
y_width_train = []
x_test = []
y_length_test = []
y_color_test = []
y_angle_test = []
y_width_test = []

for angle_iterator in range(angle):
	for length_iterator in range(length):
		for width_iterator in range(width):
			for color_iterator in range(color):
				for variation_iterator in range(variation_train):
					imgFile = str(angle_iterator) + '_' + str(length_iterator) + '_' + str(width_iterator) + '_' + str(color_iterator) + '_' + str(variation_iterator) + '.' + imgFmt
					x_train.append(cv2.imread(imgDir_train + imgFile))
					y_length_train.append(length_iterator)
					y_width_train.append(width_iterator)
					y_color_train.append(color_iterator)
					y_angle_train.append(angle_iterator)

for angle_iterator in range(angle):
	for length_iterator in range(length):
		for width_iterator in range(width):
			for color_iterator in range(color):
				for variation_iterator in range(variation_test):
					imgFile = str(angle_iterator) + '_' + str(length_iterator) + '_' + str(width_iterator) + '_' + str(color_iterator) + '_' + str(variation_iterator) + '.' + imgFmt
					x_test.append(cv2.imread(imgDir_test + imgFile))
					y_length_test.append(length_iterator)
					y_width_test.append(width_iterator)
					y_color_test.append(color_iterator)
					y_angle_test.append(angle_iterator)

np.save('x_train',np.asarray(x_train))
np.save('y_train_length',np.asarray(y_length_train))
np.save('y_train_width',np.asarray(y_width_train))
np.save('y_train_color',np.asarray(y_color_train))
np.save('y_train_angle',np.asarray(y_angle_train))

np.save('x_test',np.asarray(x_test))
np.save('y_test_length',np.asarray(y_length_test))
np.save('y_test_width',np.asarray(y_width_test))
np.save('y_test_color',np.asarray(y_color_test))
np.save('y_test_angle',np.asarray(y_angle_test))