import tensorflow as tf
import numpy as np
import landmarkGenerator

def trainLandmarkGenerator(X, Y, epoch, load=False):
	if load == False:
		model = landmarkGenerator.getModel()
	else:
		model = tf.keras.models.load_model(
			'models/landmarkGeneratorModel/model',
			compile=True
		)

	model.fit(X, Y, epochs=epoch)
	model.save('models/landmarkGeneratorModel/model')

if __name__ == '__main__':
	main()