import tensorflow as tf
import cv2
import numpy as np

imgDir = 'input_image/'
imgFmt = 'jpg'

# Format of image = angle_length_width_color_variation

angle = 12
length = 2
width = 2
color = 2
variation = 1000

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
				for variation_iterator in range(variation):
					imgFile = str(angle_iterator) + '_' + str(length_iterator) + '_' + str(width_iterator) + '_' + str(color_iterator) + '_' + str(variation_iterator) + '.' + imgFmt
					if np.any(cv2.imread(imgDir + imgFile)) != None:
						if variation_iterator > 700:
							x_test.append(cv2.imread(imgDir + imgFile))
							y_test.append(angle_iterator*8 + length_iterator*4 + width_iterator*2 + color_iterator)
						else :
							x_train.append(cv2.imread(imgDir + imgFile))
							y_train.append(angle_iterator*8 + length_iterator*4 + width_iterator*2 + color_iterator)
					count[angle_iterator*8 + length_iterator*4 + width_iterator*2 + color_iterator] = 1
				print(angle_iterator*8 + length_iterator*4 + width_iterator*2 + color_iterator)
					
np.save('x_train',np.asarray(x_train))
np.save('y_train',np.asarray(y_train))

np.save('x_test',np.asarray(x_test))
np.save('y_test',np.asarray(y_test))

for i in range(96):
	if(count[i] == 0):
		print("Wrong")
		break