import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train_length = y_train
y_train_width = y_train
y_train_color = y_train
y_train_angle = y_train

inputs = tf.keras.layers.Input(shape=(28, 28, 1), name='inputs')

feature_map = tf.keras.layers.MaxPool2D()(
	tf.keras.layers.Conv2D(32, (3, 3), activation='elu')(
		tf.keras.layers.Dropout(0.2)(
			tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(
				inputs))))

output_length = tf.keras.layers.Dense(10, activation='softmax', name='output_length')(tf.keras.layers.MaxPool2D()(tf.keras.layers.Conv2D(32, (3, 3), activation='selu')(feature_map)))
output_width = tf.keras.layers.Dense(10, activation='softmax', name='output_width')(tf.keras.layers.MaxPool2D()(tf.keras.layers.Conv2D(32, (3, 3), activation='selu')(feature_map)))
output_color = tf.keras.layers.Dense(10, activation='softmax', name='output_color')(tf.keras.layers.MaxPool2D()(tf.keras.layers.Conv2D(32, (3, 3), activation='selu')(feature_map)))
output_angle = tf.keras.layers.Dense(10, activation='softmax', name='output_angle')(tf.keras.layers.MaxPool2D()(tf.keras.layers.Conv2D(32, (3, 3), activation='selu')(feature_map)))

outputs = [output_length, output_width, output_color, output_angle]

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam',
  loss={'output_length': 'categorical_crossentropy',
   'output_width': 'categorical_crossentropy',
   'output_color': 'categorical_crossentropy',
   'output_angle': 'categorical_crossentropy'},
  metrics=['accuracy'])

print("Training###########################################################################################")
model.fit(x_train, {
	'output_length': y_train_length,
	'output_width': y_train_width,
	'output_color': y_train_color,
	'output_angle': y_train_angle
	}, epochs=5)

print("Testing############################################################################################")
result = model.evaluate(x_test, {
	'output_length': y_test_length,
	'output_width': y_test_width,
	'output_color': y_test_color,
	'output_angle': y_test_angle
	})