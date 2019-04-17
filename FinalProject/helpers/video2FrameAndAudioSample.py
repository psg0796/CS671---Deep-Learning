# from google.colab import drive
# drive.mount('/content/drive')

from moviepy.editor import *
import matplotlib.pyplot as plt
import numpy as np
# inputDir = 'drive/My Drive/FinalProject/input/'
inputDir = '../input/'
video_path = inputDir + 'videos/'
video_name = 'race-speech.mp4'
frame_output_path = inputDir + 'frames/'

video = VideoFileClip(video_path + video_name)
audio = video.audio
duration = video.duration - 2000 # == audio.duration, presented in seconds, float
#note video.fps != audio.fps

audio_frame = []
video_frame = []
step = 0.01
for t in range(int(duration / step)): # runs through audio/video frames obtaining them by timestamp with step 100 msec
    t = t * step
    if t > audio.duration or t > video.duration: break
    audio_frame.append(audio.get_frame(t)) #numpy array representing mono/stereo values
    video_frame.append(video.get_frame(t)) #numpy array representing RGB/gray frame

np.save('videoFrames/' + video_name.split(".")[0],np.asarray(video_frame))
np.save(inputDir + 'audioFrames/' + video_name.split(".")[0],np.asarray(audio_frame))