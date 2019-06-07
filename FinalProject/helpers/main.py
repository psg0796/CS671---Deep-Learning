import tensorflow as tf
import numpy as np
import landmarkGenerator
import fakeFrameGenerator

# model to generate frames from corresponding audio samples
def trainGenerator(X, Y, epoch, load=False, model=2):
	modelPath = ''
	if load == False:
		if model == 1:
			model = landmarkGenerator.getModel()
		elif model == 2:
			model = fakeFrameGenerator.getModel()
	else:
		if model == 1:
			modelPath = 'models/landmarkGeneratorModel/model'
		elif model == 2:
			modelPath = 'models/fakeFrameGenerator/model'

		model = tf.keras.models.load_model(
			modelPath,
			compile=True
		)

	model.fit(X, Y, epochs=epoch)
	model.save(modelPath)

def main():
	landmark = np.load('../input/landmark/0.npy')
	baseImage = np.load('../input/baseImage/0.npy')
	videoGt = np.load('../input/videoGt/0.npy')
	X = [landmark, baseImage]
	Y = videoGt
	EPOCH = 5
	trainGenerator(X, Y, EPOCH)

if __name__ == '__main__':
	main()