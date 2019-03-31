import tensorflow as tf
import numpy as np

filepath = 'model6/model'

model = tf.keras.models.load_model(
    filepath,
    compile=True
)
inputPath = 'inputData/'

(x_train, y_train_length, y_train_width, y_train_color, y_train_angle) = (np.load(inputPath + 'x_train.npy'), np.load(inputPath + 'y_train_length.npy'), np.load(inputPath + 'y_train_width.npy'), np.load(inputPath + 'y_train_color.npy'), np.load(inputPath + 'y_train_angle.npy'))
(x_test, y_test_length, y_test_width, y_test_color, y_test_angle) = (np.load(inputPath + 'x_test.npy'), np.load(inputPath + 'y_test_length.npy'), np.load(inputPath + 'y_test_width.npy'), np.load(inputPath + 'y_test_color.npy'), np.load(inputPath + 'y_test_angle.npy'))

model.evaluate(x_test, {
	'output_length': y_test_length,
	'output_width': y_test_width,
	'output_color': y_test_color,
	'output_angle': y_test_angle
})