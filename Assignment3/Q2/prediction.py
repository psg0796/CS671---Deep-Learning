import tensorflow as tf
import numpy as np
import os
import cv2
import glob
import math
import time

def getPadd(z, size_x, size_y):
	return ((math.floor((size_x - z[0])/2),math.ceil((size_x - z[0])/2)),(math.floor((size_y - z[1])/2),math.ceil((size_y - z[1])/2)), (0,0))

def generateNpyDataFromInput(inputDir, outputDir, inputNpyFileName, predictedCoordinatesFileName):
	x = []
	fileName = []
	size_x = 300
	size_y = 400

	files=glob.glob(inputDir + "*.*")
	
	i = 0
		
	for file in files:
		temp = cv2.imread(file)
		if temp.shape[0] > size_x or temp.shape[1] > size_y:
			print("Skipping file" + file + " because of size issues. Max size supported by model is [" + str(size_x) + "," + str(size_y) + "]")
			continue
		padding = getPadd(temp.shape, size_x, size_y)
		x.append(np.pad(temp, padding, 'constant', constant_values=(0)))
		fileName.append(file.split("/")[-1].split(".")[0] + "_gt.txt")
		print(i)
		i = i + 1

	np.save(outputDir + inputNpyFileName, np.asarray(x).reshape((-1,size_x,size_y,1)))
	np.save(outputDir + predictedCoordinatesFileName, np.asarray(fileName))

def predict(modelPath, inputDir, outputDir):
	predictionInput = 'predictionInput'
	predictedCoordinatesFileName = 'predictedCoordinatesFileName'

	os.popen('rm -rf ' + outputDir)
	time.sleep(1)
	os.popen('mkdir ' + outputDir)
	time.sleep(1)
	os.popen('mkdir ' + outputDir + 'predictedVals')
	generateNpyDataFromInput(inputDir, outputDir, predictionInput, predictedCoordinatesFileName)

	inputData = np.load(outputDir + predictionInput + '.npy')
	predictedCoordinatesFileName = np.load(outputDir + predictedCoordinatesFileName + '.npy')

	model = tf.keras.models.load_model(
	    modelPath,
	    compile=True
	)

	i = 0
	for x in inputData:
		os.popen("echo \"" + str(model.predict(inputData[i:i+1])) + "\" > " + outputDir + "predictedVals/" + str(predictedCoordinatesFileName[i]))
		i = i + 1

if __name__ == '__main__':
	inputDir = input("Enter the input diretory path : ")
	predict('assignment_3_model', inputDir, "predictionOutput/")