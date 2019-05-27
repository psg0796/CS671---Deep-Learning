import tensorflow as tf
import numpy as np
import os
import dataGenerator
import sys, getopt
import prediction
import time

localDataDir = "localData/"
inputNpyFileName = "input"
outputLabelsNpyFileName = "labels"

def createModel(size_x, size_y):
	inputs = tf.keras.layers.Input(shape=(size_x, size_y, 1), name='inputs')

	feature_map = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(
	    tf.keras.layers.Conv2D(32, (5, 5), activation='relu')(
	      tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(
	        tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(
	          inputs))))

	output = tf.keras.layers.Dense(2, activation='sigmoid', name='output')(tf.keras.layers.Dense(128, activation='relu')(tf.keras.layers.Dropout(0.2)(tf.keras.layers.Dense(256, activation='relu')(tf.keras.layers.Dense(256, activation='relu')(tf.keras.layers.Flatten()(feature_map))))))

	outputs = [output]

	model = tf.keras.Model(inputs=inputs, outputs=outputs)

	model.compile(optimizer='adam',
	  loss={'output': 'squared_hinge'},
	  metrics=['accuracy'])

	model.save(localDataDir + 'model')

def training(x, y, epoch):
	print("#############################################				Training				##############################################")
	if(len(os.popen("ls " + localDataDir + "| grep model").readlines()) == 0):
		max_img_dim = np.load(localDataDir + 'max_img_dim.npy')
		createModel(max_img_dim[0], max_img_dim[1])

	model = tf.keras.models.load_model(
	    localDataDir + 'model',
	    compile=True
	)

	model.fit(x, {
		'output': y
	}, epochs=epoch)

def testing(x, y):
	print("#############################################				Testing					###############################################")
	
	model = tf.keras.models.load_model(
	    localDataDir + 'model',
	    compile=True
	)

	model.evaluate(x, {
		'output': y,
		})

def main(argv):
	opts, args = getopt.getopt(argv,"hp:e:",["help","phase=","epochs="])
	for opt, arg in opts:
		if opt == '-h' or opt == '--help':
			print('test.py -p <phase> -e <epoch>')
			print('test.py --phase=<phase> --epochs=<epoch>')
			print('phase can be "training" or "testing" or "predict"')
			sys.exit()
		elif opt in ("-p", "--phase"):
			phase = arg
		elif opt in ("-e", "--epoch"):
			epoch = int(arg)
	
	inputDir = input("Enter the input diretory path : ")

	if phase == "training":
		os.popen("rm -rf " + localDataDir)
		os.popen("mkdir " + localDataDir)
		time.sleep(1)
		dataGenerator.generateNpyDataFromInput(inputDir, localDataDir, inputNpyFileName, outputLabelsNpyFileName)
		x = np.load(localDataDir + inputNpyFileName + ".npy")
		y = np.load(localDataDir + outputLabelsNpyFileName + ".npy")
		training(x, y, epoch)

	elif phase == "testing":
		dataGenerator.generateNpyDataFromInput(inputDir, localDataDir, inputNpyFileName, outputLabelsNpyFileName)
		x = np.load(localDataDir + inputNpyFileName + ".npy")
		y = np.load(localDataDir + outputLabelsNpyFileName + ".npy")
		testing(x, y)

	elif phase == "predict":
		os.popen("mkdir " + localDataDir + "prediction")
		time.sleep(1)
		prediction.predict(localDataDir + 'model', inputDir, localDataDir + 'prediction/', localDataDir)
	
if __name__ == '__main__':
	if(len(sys.argv) == 1):
		print("try using \"-h\" or \"--help\"")
		sys.exit()
	main(sys.argv[1:])