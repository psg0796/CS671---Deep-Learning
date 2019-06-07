import tensorflow as tf
import numpy as np

def getModel():
	model = tf.keras.Sequential()
	model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(2,)))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Reshape((7, 7, 256)))

	model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	return model