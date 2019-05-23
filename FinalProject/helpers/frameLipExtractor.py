#!/usr/bin/env python
# coding: utf-8

# In[90]:


#!/usr/bin/env python
# coding: utf-8

# In[173]:


import cv2
import numpy as np
import dlib
import numpy
from PIL import Image, ImageDraw

presentDir = ''
predictor = dlib.shape_predictor(presentDir + "shape_predictor_68_face_landmarks copy.dat")

def findBoundingBox(mask):
	x1,y1 = 1280,720
	x2,y2 = -1,-1
	for point in mask:
		x1,y1 = min(x1,point[0]), min(y1,point[1])
		x2,y2 = max(x2,point[0]), max(y2,point[1])

	return x1,x2,y1,y2


def lipExtractor(inputNpy):
	gray = inputNpy

	detector = dlib.get_frontal_face_detector()
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
		q  = list(zip(s,t))
		
		x1,x2,y1,y2 = findBoundingBox(q)
		
		# read image as RGB and add alpha (transparency)
		im = Image.fromarray(gray).convert('RGBA')

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
		newImArray = np.copy(inputNpy)
		
		for i in range(3):
			newImArray[:,:,i] = newImArray[:,:,i] * mask

		crop = newImArray[y1:y2,x1:x2]
		
		return crop,False
	return None,True
