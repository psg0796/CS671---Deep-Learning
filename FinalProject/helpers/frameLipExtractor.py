import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
import numpy
from PIL import Image, ImageDraw

outputDir = '../input/lipVideoFrames/'

def outputGenerator(input = "/home/psg/Desktop/Screenshot from 2019-05-04 18-25-26.png"):
	gray = cv2.imread(input)

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks copy.dat")
	faces = detector(gray)
	for face in faces:
	    x1 = face.left()
	    y1 = face.top()
	    x2 = face.right()
	    y2 = face.bottom()
	landmarks = predictor(gray, face)
	a = []
	b = []
	s = []
	t = []
	for n in range(48, 60):
	    a= []
	    x = landmarks.part(n).x
	    y = landmarks.part(n).y
	    a.append(x)
	    s.append(x)
	    t.append(y)
	    a.append(y)
	    b.append(a)
	c = np.array(b).reshape((-1,1,2))
	cv2.polylines(gray,[c],True,(0,255,255))
	# cv2.imwrite("rohan.png",gray)
	q  = list(zip(s,t))
	print(q)

	# read image as RGB and add alpha (transparency)
	im = Image.open(input).convert("RGBA")

	# convert to numpy (for convenience)
	imArray = numpy.asarray(im)

	# create mask
	polygon = [(4,203),(6,243),(69,177),(58,26),(42,42)]
	polygon = q
	maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
	ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
	mask = numpy.array(maskIm)

	# assemble new image (uint8: 0-255)
	newImArray = numpy.empty(imArray.shape,dtype='uint8')

	# colors (three first columns, RGB)
	newImArray[:,:,:3] = imArray[:,:,:3]

	# transparency (4th column)
	newImArray[:,:,3] = mask*255

	# back to Image from numpy
	newIm = Image.fromarray(newImArray, "RGBA")
	
	newIm.save(outputDir + input.split("/")[-1])

if __name__ == '__main__':
	outputGenerator()