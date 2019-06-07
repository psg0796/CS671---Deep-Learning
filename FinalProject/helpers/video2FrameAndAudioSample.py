from moviepy.editor import *
import numpy as np
import glob
import utils

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
		landmarks = []
		step = 10
		preT = 0
		for t in range(int(duration / step)): # runs through audio/video frames obtaining them by timestamp with step 100 msec
			t = t * step
			print(t)
			if(t > audio.duration or t > video.duration): break
			landmarks.append(utils.get_landmark(video.get_frame(t)))
			audio_frame.append(audio.get_frame(t)) #numpy array representing mono/stereo values
			video_frame.append(video.get_frame(t)) #numpy array representing RGBA frame
			if(preT != int(t)):
				preT = int(t)
				np.save(inputDir + 'landmark/' + str(videoCount),np.asarray(landmarks))
				np.save(inputDir + 'videoGt/' + str(videoCount),np.asarray(video_frame))
				np.save(inputDir + 'audioGt/' + str(videoCount),np.asarray(audio_frame))
				audio_frame = []
				video_frame = []
				landmarks = []
				videoCount = videoCount + 1
				print("########################" + str(videoCount) + "########################")

if __name__ == '__main__':
	video2frame()
