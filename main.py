import sys
from Detect import Detector
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
from filters import filter_image, warp
from util import vconcat_resize, hconcat_resize

detector = Detector()
mode = sys.argv[2]


def process_image(image):
    filtered_binary = filter_image(image)
    binary_warped = warp(filtered_binary * 255)
    b_img = np.dstack(
        (binary_warped, binary_warped, binary_warped))*255
    f_image = np.dstack(
        (filtered_binary, filtered_binary, filtered_binary))*255
    result, out_img = detector.detect_lanes(binary_warped, image)
    if mode == '-d' or mode == '--debug':
        return vconcat_resize([result, hconcat_resize([b_img, f_image, out_img])])
    elif mode == '-p' or mode == '--prod':
        return result
    else:
        return result


if __name__ == '__main__':
    input_path = sys.argv[1]
    white_output = 'challenge_video_out.mp4'
    clip1 = VideoFileClip(input_path)
    # NOTE: this function expects color images!!
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)
