import numpy as np
import matplotlib.pyplot as plt

inputFiles = ['cummulative_activation_task1.npy', 'cummulative_activation_task2.npy']
outputDir = [ ['ActivationImages/task1/img1', 'ActivationImages/task1/img2', 'ActivationImages/task1/img3', 'ActivationImages/task1/img4', 'ActivationImages/task1/img5', 'ActivationImages/task1/img6', ], ['ActivationImages/task2/img1', 'ActivationImages/task2/img2', 'ActivationImages/task2/img3', 'ActivationImages/task2/img4', 'ActivationImages/task2/img5', 'ActivationImages/task2/img6'] ]
layers = [ [1], [1] ]
numOfImages = [ [16], [32] ]

for fileIndex in range(len(inputFiles)):
	for field in range(len(outputDir[fileIndex])):
		for layerIndex in range(len(layers[fileIndex])):
			img = np.load(inputFiles[fileIndex])
			for imageLayer in range(numOfImages[fileIndex][layerIndex]):
				print(field)
				print(layers[fileIndex][layerIndex])
				print(imageLayer)
				print("#####################################################")
				plt.imshow(img[field][layers[fileIndex][layerIndex]][0][:,:,imageLayer])
				plt.savefig(outputDir[fileIndex][field] + '/' + str(imageLayer))