# from moviepy.editor import *
import numpy as np
import glob
import cv2
import utils

inputDir = '../input/'
video_path = inputDir + 'videos/'

def video2frame():
	videos = glob.glob(video_path + "*")
	videoCount = 0
	for video in videos:
		vidObj = cv2.VideoCapture(video)
		success = 1
		count = 0

		# audio_frame = []
		video_frame = []
		landmarks = []
			
		while success:
			success, image = vidObj.read()
			if success != 1:
				break
			if count == 1000:
				break
			count += 1
			frame_output_path = inputDir + 'frames/'
			# video = VideoFileClip(video)
			# audio = video.audio
			# duration = video.duration # == audio.duration, presented in seconds, float
			# #note video.fps != audio.fps

			# step = 10
			# preT = 0
			# for t in range(int(duration / step)): # runs through audio/video frames obtaining them by timestamp with step 100 msec
			# 	t = t * step
			# 	print(t)
			# 	if(t > audio.duration or t > video.duration): break
			landmarks.append(utils.get_landmark(image))
				# audio_frame.append(audio.get_frame(t)) #numpy array representing mono/stereo values
			video_frame.append(image) #numpy array representing RGBA frame
			print(count)
				# if(preT != int(t)):
				# 	preT = int(t)
		
		np.save(inputDir + 'landmark/' + str(videoCount),np.asarray(landmarks))
		np.save(inputDir + 'videoGt/' + str(videoCount),np.asarray(video_frame))
		videoLen = len(video_frame)
		middleFrame = video_frame[videoLen//2]
		video_frame = []
		landmarks = []
		baseImage = []
		for i in range(videoLen):
			baseImage.append(middleFrame)
		np.save(inputDir + 'baseImage/' + str(videoCount),np.asarray(baseImage))
		baseImage = []
					# np.save(inputDir + 'audioGt/' + str(videoCount),np.asarray(audio_frame))
					# audio_frame = []
					# video_frame = []
					# landmarks = []
		videoCount = videoCount + 1
		print("########################" + str(videoCount) + "########################")

if __name__ == '__main__':
	video2frame()
