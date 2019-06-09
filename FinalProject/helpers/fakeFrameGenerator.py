import tensorflow as tf
import numpy as np

def getModel():
	landmark = tf.keras.layers.Input(shape=(240, 320, 3), name='landmark')
	baseImage = tf.keras.layers.Input(shape=(240, 320, 6), name='baseImage')

	conv2D_pre_1 = tf.keras.layers.Conv2D(128, (7, 7), activation='relu')(baseImage)
	maxpool_pre_1 = tf.keras.layers.MaxPool2D(pool_size=(5, 5))(conv2D_pre_1)
	conv2D_pre_2 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu')(maxpool_pre_1)
	batchNormalization_pre_1 = tf.keras.layers.BatchNormalization()(conv2D_pre_2)
	conv2D_pre_3 = tf.keras.layers.Conv2D(16, (5, 5), activation='relu')(batchNormalization_pre_1)
	conv2D_pre_4 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu')(conv2D_pre_3)
	
	conv2DTrans_pre_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3),activation='relu')(conv2D_pre_4)
	conv2DTrans_pre_2 = tf.keras.layers.Conv2DTranspose(64, (5, 5),activation='relu')(conv2DTrans_pre_1)
	conv2DTrans_pre_3 = tf.keras.layers.Conv2DTranspose(8, (7, 7), activation='relu')(conv2DTrans_pre_2)
	upsample_pre_1 = tf.keras.layers.UpSampling2D(size=(5, 5))(conv2DTrans_pre_3)
	conv2DTrans_pre_4 = tf.keras.layers.Conv2DTranspose(3, (1, 1), activation='relu')(upsample_pre_1)
	
	concat = tf.keras.layers.concatenate([landmark, conv2DTrans_pre_4])
	conv2D_1 = tf.keras.layers.Conv2D(128, (5, 5), activation='relu')(concat)
	maxpool2D_1 = tf.keras.layers.MaxPool2D(pool_size=(5, 5))(conv2D_1)
	batchNormalization_1 = tf.keras.layers.BatchNormalization()(maxpool2D_1)
	conv2D_2 = tf.keras.layers.Conv2D(64, (5, 5))(batchNormalization_1)
	leakyRelu_1 = tf.keras.layers.LeakyReLU()(conv2D_2)
	conv2D_3 = tf.keras.layers.Conv2D(16, (5, 5))(leakyRelu_1)

	conv2DTrans_1 = tf.keras.layers.Conv2DTranspose(3, (5, 5))(conv2D_3)
	batchNormalization_2 = tf.keras.layers.BatchNormalization()(conv2DTrans_1)
	conv2DTrans_2 = tf.keras.layers.Conv2DTranspose(3, (5, 5))(batchNormalization_2)
	upsample_1 = tf.keras.layers.UpSampling2D(size=(5, 5))(conv2DTrans_2)
	dropOut_1 = tf.keras.layers.Dropout(0.5)(upsample_1)
	conv2DTrans_3 = tf.keras.layers.Conv2DTranspose(3, (6, 6), activation='sigmoid')(dropOut_1)
	
	model = tf.keras.Model(inputs = [landmark, baseImage], outputs = conv2DTrans_3)

	model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

	return model