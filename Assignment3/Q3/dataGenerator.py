import tensorflow as tf
import cv2
import numpy as np
import glob
import math
import os

def getPadd(z, size_x, size_y):
	return ((math.floor((size_x - z[0])/2),math.ceil((size_x - z[0])/2)),(math.floor((size_y - z[1])/2),math.ceil((size_y - z[1])/2)))

def generateNpyDataFromInput(inputDir, outputDir, inputNpyFileName, outputLabelsNpyFileName):
	imgDir = 'Data/'
	groundTruthDir = 'Ground_truth/'
	x = []
	labels = []
	size_x = 0
	size_y = 0

	files=glob.glob(inputDir + imgDir + "*.*")
	i = 0
	
	for file in files:
		temp = cv2.imread(file, 0)
		size_x = max(size_x, temp.shape[0])
		size_y = max(size_y, temp.shape[1])
	
	np.save(outputDir + 'max_img_dim', np.asarray([size_x, size_y]))
	
	for file in files:
		temp = cv2.imread(file, 0)
		padding = getPadd(temp.shape, size_x, size_y)
		x.append(np.pad(temp, padding, 'constant', constant_values=(0)))
		f = os.popen("cat " + inputDir + groundTruthDir + file.split("/")[-1].split(".")[0] + "*")
		val = f.readline().split(" ")
		labels.append([padding[0][0] + int(val[0]), padding[1][0] + int(val[1])])
		print(i)
		i = i + 1

	np.save(outputDir + inputNpyFileName, np.asarray(x).reshape((-1,size_x,size_y,1)))
	np.save(outputDir + outputLabelsNpyFileName, np.asarray(labels))