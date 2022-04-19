from Detect import Detector
from moviepy.editor import VideoFileClip
from filters import filter_image, warp

detector = Detector()


def process_image(image):
    filtered_binary = filter_image(image)
    binary_warped = warp(filtered_binary)
    final_image = detector.detect_lanes(binary_warped, image)
    return final_image


if __name__ == '__main__':
    white_output = 'challenge_video_out.mp4'
    clip1 = VideoFileClip("videos/challenge_video.mp4")
    # NOTE: this function expects color images!!
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)
