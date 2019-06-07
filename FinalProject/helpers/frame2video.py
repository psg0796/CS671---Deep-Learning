import cv2
import numpy as np
import glob
from PIL import Image

numOfVideoNpyFiles = 18

def convert_frames_to_video(pathIn,pathOut,fps):
	frame_array = []
	videos = glob.glob(pathIn + "*")
	for i in range(numOfVideoNpyFiles + 1):
		print(i)
		videoFrames = np.load(pathIn + str(i) + '.npy')
		for oimg in videoFrames:
			Image.fromarray(oimg).save('temp.jpeg')
			img = cv2.imread('temp.jpeg')
			height, width, layers = img.shape
			size = (width,height)
			frame_array.append(img)
 	
	out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

	for i in range(len(frame_array)):
		out.write(frame_array[i])
	out.release()

def main():
    pathIn= 'vf/'
    pathOut = 'video.avi'
    fps = 25.0
    convert_frames_to_video(pathIn, pathOut, fps)
 
if __name__=="__main__":
    main()