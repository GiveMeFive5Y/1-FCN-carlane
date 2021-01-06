import numpy as np
import cv2
from scipy.misc import imresize
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import tensorflow as tf
import matplotlib.pyplot as plt


# class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def road_lines(image):
    small_img = scipy.misc.toimage(scipy.misc.imresize(image,(160,576,3)))
    small_img = small_img[None,:,:,:]

    # Make prediction with nerual network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255
    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)

    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions ,stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)

    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = imresize(lane_drawn,(720,1280,3))

    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1 ,lane_image, 1, 0)

    return result

# load model
model = tf.train.import_meta_graph(sess, ['model.pb.meta'])
with tf.Session() as sess:
    model.restore(sess, )
    lanes = Lanes()
    output = ''
    clip1 = VideoFileClip('')
    vid_clip = clip1.fl_image(road_lines)
    vid_clip.wriet_videofile(output,audio=False)