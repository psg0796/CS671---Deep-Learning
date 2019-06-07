from imutils import face_utils
import dlib
import cv2
import numpy as np

# from gtts import gTTS 
# import os 

# Source get_landmark() :- https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672
# Source text2speech() :- geeksforgeeks

# Vamos inicializar um detector de faces (HOG) para então
# let's go code an faces detector(HOG) and after detect the 
# landmarks on this detected face

# p = our pre-treined model directory, on my case, it's on the same script's diretory.
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

def get_landmark(inputImg):
	# Getting out image by webcam 
	image = inputImg
	output = np.ones(image.shape)
	# Converting the image to gray scale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# Get faces into webcam's image
	rects = detector(gray, 0)

	# For each detected face, find the landmark.
	for (i, rect) in enumerate(rects):
		# Make the prediction and transfom it to numpy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# Draw on our image, all the finded cordinate points (x,y) 
		for (x, y) in shape:
			cv2.circle(output, (x, y), 2, (0, 255, 0), -1)

	return output

# def text2speech(mytext):
# 	# Language in which you want to convert 
# 	language = 'en'

# 	# Passing the text and language to the engine,  
# 	# here we have marked slow=False. Which tells  
# 	# the module that the converted audio should  
# 	# have a high speed 
# 	myobj = gTTS(text=mytext, lang=language, slow=False) 

# 	myobj.save("Hello.mp3") 

def video2landmarkNpy(videoPath):
	vidObj = cv2.VideoCapture(videoPath)
		success = 1
		count = 0
		landmarks = []
			
		while success:
			success, image = vidObj.read()
			if success != 1:
				break
			count += 1
			print(count)
			landmarks.append(get_landmark(image))

	return np.asarray(landmarkSequence)