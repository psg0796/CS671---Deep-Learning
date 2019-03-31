import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

model_dir = 'model7'
os.mkdir(model_dir)
epoch = 20
model_name = 'epoch_' + str(epoch) + '_conv(32,(3,3),relu)_batchNorm_maxPool((2,2),2)_(relu,relu,relu,relu)_(128,128,128,256)_(sigmoid,sigmoid,sigmoid,softmax).png'
inputPath = 'inputData/'

def plot_history(history):
	hist = pd.DataFrame(history.history)
	hist['epoch'] = history.epoch
	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.plot(hist['epoch'], hist['output_length_loss'],
	       label='Length Loss')
	plt.plot(hist['epoch'], hist['output_width_loss'],
	       label='Width Loss')
	plt.plot(hist['epoch'], hist['output_color_loss'],
	       label='Color Loss')
	plt.plot(hist['epoch'], hist['output_angle_loss'],
	       label='Angle Loss')
	plt.plot(hist['epoch'], hist['loss'],
	       label='Train Loss')
	plt.ylim([0,1])
	plt.legend()
	plt.savefig(model_dir+'/loss_plot_'+model_name)

	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')

	plt.plot(hist['epoch'], hist['output_length_acc'],
	       label='Length Acc')
	plt.plot(hist['epoch'], hist['output_width_acc'],
	       label='Width Acc')
	plt.plot(hist['epoch'], hist['output_color_acc'],
	       label='Color Acc')
	plt.plot(hist['epoch'], hist['output_angle_acc'],
	       label='Angle Acc')
	plt.ylim([0,1])
	plt.legend()
	plt.savefig(model_dir+'/acc_plot_'+model_name)

################################ For MNIST dataset ##########################################################
# 																											#
# 	mnist = tf.keras.datasets.mnist 																		#
#	(x_train, y_train),(x_test, y_test) = mnist.load_data()													#
# 	x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)													#
# 	x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)														#
# 	x_train, x_test = x_train / 255.0, x_test / 255.0														#
# 																											#
# 	y_train_length = y_train 																				#
# 	y_train_width = y_train 																				#
# 	y_train_color = y_train 																				#
# 	y_train_angle = y_train 																				#
# 	y_test_length = y_train 																				#
# 	y_test_width = y_train 																					#
# 	y_test_color = y_train 																					#
# 	y_test_angle = y_train 																					#
#																											#
#############################################################################################################

(x_train, y_train_length, y_train_width, y_train_color, y_train_angle) = (np.load(inputPath + 'x_train.npy'), np.load(inputPath + 'y_train_length.npy'), np.load(inputPath + 'y_train_width.npy'), np.load(inputPath + 'y_train_color.npy'), np.load(inputPath + 'y_train_angle.npy'))
(x_test, y_test_length, y_test_width, y_test_color, y_test_angle) = (np.load(inputPath + 'x_test.npy'), np.load(inputPath + 'y_test_length.npy'), np.load(inputPath + 'y_test_width.npy'), np.load(inputPath + 'y_test_color.npy'), np.load(inputPath + 'y_test_angle.npy'))

inputs = tf.keras.layers.Input(shape=(28, 28, 3), name='inputs')

feature_map = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(
	tf.keras.layers.BatchNormalization(axis = 3)(
		tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(
			inputs)))

output_length = tf.keras.layers.Dense(2, activation='sigmoid', name='output_length')(tf.keras.layers.Dropout(0.2)(tf.keras.layers.Dense(128, activation='relu')(tf.keras.layers.Flatten()(feature_map))))
output_width = tf.keras.layers.Dense(2, activation='sigmoid', name='output_width')(tf.keras.layers.Dropout(0.2)(tf.keras.layers.Dense(128, activation='relu')(tf.keras.layers.Flatten()(feature_map))))
output_color = tf.keras.layers.Dense(2, activation='sigmoid', name='output_color')(tf.keras.layers.Dropout(0.2)(tf.keras.layers.Dense(128, activation='relu')(tf.keras.layers.Flatten()(feature_map))))
output_angle = tf.keras.layers.Dense(12, activation='softmax', name='output_angle')(tf.keras.layers.Dropout(0.2)(tf.keras.layers.Dense(256, activation='relu')(tf.keras.layers.Flatten()(feature_map))))

outputs = [output_length, output_width, output_color, output_angle]

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam',
  loss={'output_length': 'sparse_categorical_crossentropy',
   'output_width': 'sparse_categorical_crossentropy',
   'output_color': 'sparse_categorical_crossentropy',
   'output_angle': 'sparse_categorical_crossentropy'},
  metrics=['accuracy'])

print("#############################################				Training				##############################################")
model_history = model.fit(x_train, {
	'output_length': y_train_length,
	'output_width': y_train_width,
	'output_color': y_train_color,
	'output_angle': y_train_angle
	}, epochs=epoch)

plot_history(model_history)

print("#############################################				Testing					###############################################")
result = model.evaluate(x_test, {
	'output_length': y_test_length,
	'output_width': y_test_width,
	'output_color': y_test_color,
	'output_angle': y_test_angle
	})

tf.keras.utils.plot_model(
    model,
    to_file=model_dir+'/model_'+model_name,
    show_shapes=False,
    show_layer_names=True,
    rankdir='TB'
)

model.save(model_dir + '/model')

print("###############################          END				###################################################")

f = open(model_dir + '/result', 'a')
f.write(str(result))
f.close()
