from moviepy.editor import *
#import matplotlib.pyplot as plt
import numpy as np
import glob
import frameLipExtractor

inputDir = '../input/'
video_path = inputDir + 'videos/'

def video2frame():
	videos = glob.glob(video_path + "*")
	videoCount = 0
	for video in videos:
		frame_output_path = inputDir + 'frames/'
		video = VideoFileClip(video)
		audio = video.audio
		duration = video.duration # == audio.duration, presented in seconds, float
		#note video.fps != audio.fps

		audio_frame = []
		video_frame = []
		step = 0.01
		preT = 0
		for t in range(int(duration / step)): # runs through audio/video frames obtaining them by timestamp with step 100 msec
			t = t * step
			print(t)
			if(t > audio.duration or t > video.duration): break
			lipFrame,err = frameLipExtractor.lipExtractor(video.get_frame(t))
			if(err != True):
				audio_frame.append(audio.get_frame(t)) #numpy array representing mono/stereo values
				video_frame.append(lipFrame) #numpy array representing RGBA frame
				if(preT != int(t)):
					preT = int(t)
					np.save(inputDir + 'videoFrames/' + str(videoCount),np.asarray(video_frame))
					np.save(inputDir + 'audioFrames/' + str(videoCount),np.asarray(audio_frame))
					audio_frame = []
					video_frame = []
					videoCount = videoCount + 1
					print("########################" + str(videoCount) + "########################")

if __name__ == '__main__':
	video2frame()
