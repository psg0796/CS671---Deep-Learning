import tensorflow as tf
import cv2
import numpy as np

imgDir1 = '../input1/'
imgDir2 = '../input2/'
imgFmt = 'jpg'

# Format of image = angle_length_width_color_variation

angle = 12
length = 2
width = 2
color = 2
variation1 = 600
variation2 = 400

x_train = []
y_train = []
x_test = []
y_test = []

count = []
for i in range(96):
	count.append(0)

for angle_iterator in range(angle):
	for length_iterator in range(length):
		for width_iterator in range(width):
			for color_iterator in range(color):
				for variation_iterator in range(variation1):
					imgFile = str(angle_iterator) + '_' + str(length_iterator) + '_' + str(width_iterator) + '_' + str(color_iterator) + '_' + str(variation_iterator) + '.' + imgFmt
					x_train.append(cv2.imread(imgDir1 + imgFile))
					y_train.append(angle_iterator*8 + length_iterator*4 + width_iterator*2 + color_iterator)
				print(angle_iterator*8 + length_iterator*4 + width_iterator*2 + color_iterator)

for angle_iterator in range(angle):
	for length_iterator in range(length):
		for width_iterator in range(width):
			for color_iterator in range(color):
				for variation_iterator in range(variation2):
					imgFile = str(angle_iterator) + '_' + str(length_iterator) + '_' + str(width_iterator) + '_' + str(color_iterator) + '_' + str(variation_iterator) + '.' + imgFmt
					x_test.append(cv2.imread(imgDir2 + imgFile))
					y_test.append(angle_iterator*8 + length_iterator*4 + width_iterator*2 + color_iterator)
				print(angle_iterator*8 + length_iterator*4 + width_iterator*2 + color_iterator)

np.save('x_train',np.asarray(x_train))
np.save('y_train',np.asarray(y_train))

np.save('x_test',np.asarray(x_test))
np.save('y_test',np.asarray(y_test))
