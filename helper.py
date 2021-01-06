import numpy as np
import cv2
from scipy.misc import imresize
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import tensorflow as tf
import matplotlib.pyplot as plt

#class to average lanes with
class Lanes():
	def __init__(self):
		self.recent_fit = []
		self.avg_fit = []

def road_lines(image):
	small_img = scipy.misc.

#load model
model = tf.train.import_meta_graph(sess,['model.pb.meta'])
with tf.Session() as sess:
	model.restore(sess,)