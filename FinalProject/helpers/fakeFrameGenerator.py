import tensorflow as tf
import numpy as np

def getModel():
	landmark = tf.keras.layers.Input(shape=(240, 320, 3), name='landmark')
	baseImage = tf.keras.layers.Input(shape=(240, 320, 3), name='baseImage')

	concat = tf.keras.layers.concatenate([landmark, baseImage])
	conv2D_1 = tf.keras.layers.Conv2D(128, (5, 5), activation='relu')(concat)
	batchNormalization_1 = tf.keras.layers.BatchNormalization()(conv2D_1)
	conv2D_2 = tf.keras.layers.Conv2D(64, (5, 5))(batchNormalization_1)
	leakyRelu_1 = tf.keras.layers.LeakyReLU()(conv2D_2)
	conv2D_3 = tf.keras.layers.Conv2D(1, (5, 5))(leakyRelu_1)

	conv2DTrans_1 = tf.keras.layers.Conv2DTranspose(3, (5, 5))(leakyRelu_1)
	batchNormalization_2 = tf.keras.layers.BatchNormalization()(conv2DTrans_1)
	conv2DTrans_2 = tf.keras.layers.Conv2DTranspose(3, (5, 5))(batchNormalization_2)
	dropOut_1 = tf.keras.layers.Dropout(0.5)(conv2DTrans_2)
	relu_1 = tf.keras.layers.ReLU()(dropOut_1)

	model = tf.keras.Model(inputs = [landmark, baseImage], outputs = relu_1)

	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	return model