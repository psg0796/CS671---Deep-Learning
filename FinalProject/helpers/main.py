import tensorflow as tf
import numpy as np
import landmarkGenerator
import fakeFrameGenerator

# model to generate frames from corresponding audio samples
def trainGenerator(X, Y, epoch, load=0, modelType=2, modelPath=''):
	modelDir = ''

	if modelType == 1:
		modelDir = 'models/landmarkGeneratorModel/'
	elif modelType == 2:
		modelDir = 'models/fakeFrameGenerator/'

	if load == 0:
		if modelType == 1:
			model = landmarkGenerator.getModel()
		elif modelType == 2:
			model = fakeFrameGenerator.getModel()
	else:
		# model = tf.keras.models.load_model(
		# 	modelPath,
		# 	compile=True
		# )
		model = tf.contrib.saved_model.load_keras_model(modelPath)
		adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)
		model.compile(optimizer=adam, loss='mean_absolute_error', metrics=['accuracy'])

	Y1 = model.predict(X)
	np.save('testResults', (Y1*255).astype(int))
	np.save('testTargets', (Y*255).astype(int))
	# model.fit(X, Y, epochs=epoch)
	# # model.save(modelPath)
	# tf.contrib.saved_model.save_keras_model(model, modelDir)

def main():
	load = int(input('Load model '))
	modelType = 2 #input('1: landmarkGenerator, 2: fakeFrameGenerator')
	landmark = np.load('../input/landmark/0.npy')/255.0
	baseImage = np.load('../input/baseImage/0.npy')/255.0
	videoGt = np.load('../input/videoGt/0.npy')/255.0
	X = {'landmark': landmark, 'baseImage': baseImage}
	Y = videoGt
	EPOCH = 5
	modelPath = 'models/fakeFrameGenerator/R10/'
	trainGenerator(X, Y, EPOCH, load, modelType, modelPath)

if __name__ == '__main__':
	main()