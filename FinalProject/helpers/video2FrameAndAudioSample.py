# from google.colab import drive
# drive.mount('/content/drive')

from moviepy.editor import *
import matplotlib.pyplot as plt
import numpy as np
import glob

# inputDir = 'drive/My Drive/FinalProject/input/'
inputDir = '../input/'
video_path = inputDir + 'videos/'

def video2frame():
	videos = glob.glob(video_path + "*")
	videoCount = 0
	for video_name in videos:
		frame_output_path = inputDir + 'frames/'
		video = VideoFileClip(video_path + video_name)
		audio = video.audio
		duration = video.duration # == audio.duration, presented in seconds, float
		#note video.fps != audio.fps

		audio_frame = []
		video_frame = []
		step = 0.01
		for t in range(int(duration / step)): # runs through audio/video frames obtaining them by timestamp with step 100 msec
		    t = t * step
		    print(t)
		    if t > audio.duration or t > video.duration: break
		    audio_frame.append(audio.get_frame(t)) #numpy array representing mono/stereo values
		    video_frame.append(video.get_frame(t)) #numpy array representing RGB/gray frame

		np.save(inputDir + 'videoFrames/' + str(videoCount),np.asarray(video_frame))
		np.save(inputDir + 'audioFrames/' + str(videoCount),np.asarray(audio_frame))
		videoCount = videoCount + 1

if __name__ == '__main__':
	video2frame()