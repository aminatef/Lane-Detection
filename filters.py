#from moviepy.editor import VideoFileClip
#from IPython.display import HTML
import numpy as np
import cv2
import pickle
import glob
#from tracker import tracker
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

project_vid = "Project_data/project_video.mp4"
challenge_vid = "Project_data/challenge_video.mp4"
evil_vid = "Project_data/harder_challenge_video.mp4"
vid = [project_vid, challenge_vid, evil_vid]
test_file = "Project_data/test_images/"
test_images = ["straight_lines1.jpg", "straight_lines2.jpg", "test1.jpg",
               "test2.jpg", "test3.jpg", "test4.jpg", "test5.jpg", "test6.jpg"]
test_images = [test_file+ele for ele in test_images]


def show_vid(name, func):
    capture = cv2.VideoCapture(name)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)+0.5)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)+0.5)
    frame_size = (width, height)
    output = cv2.VideoWriter('out_edge.mp4', cv2.VideoWriter_fourcc(
        *'mp4v'), 25, frame_size, isColor=False)
    if (capture.isOpened() == False):
        print("Error opening video  file")
    while(capture.isOpened()):
        return_, frame = capture.read()
        if return_:
            cv2.imshow("Frame", 255*func(frame))
            # cv2.imshow("Frame",frame)
            # output.write(255*func(frame))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    capture.release()
    cv2.destroyAllWindows()


def abs_sobel_(img, min_thresh=25, max_thresh=255, sobel_kernel=3):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    #hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    abs_sobel_h_channel = np.absolute(
        cv2.Sobel(h_channel, cv2.CV_64F, 1, 1, ksize=sobel_kernel))
    abs_sobel_s_channel = np.absolute(
        cv2.Sobel(s_channel, cv2.CV_64F, 1, 1, ksize=sobel_kernel))
    abs_sobel_l_channel = np.absolute(
        cv2.Sobel(l_channel, cv2.CV_64F, 1, 1, ksize=sobel_kernel))

    scaled_sobel_s = np.uint8(
        255*abs_sobel_s_channel/np.max(abs_sobel_s_channel))
    binary_output_s = np.zeros_like(scaled_sobel_s)
    binary_output_s[(scaled_sobel_s >= min_thresh) &
                    (scaled_sobel_s <= max_thresh)] = 1

    scaled_sobel_l = np.uint8(
        255*abs_sobel_l_channel/np.max(abs_sobel_l_channel))
    binary_output_l = np.zeros_like(scaled_sobel_l)
    binary_output_l[(scaled_sobel_l >= min_thresh) &
                    (scaled_sobel_l <= max_thresh)] = 1

    scaled_sobel_h = np.uint8(
        255*abs_sobel_h_channel/np.max(abs_sobel_h_channel))
    binary_output_h = np.zeros_like(scaled_sobel_h)
    binary_output_h[(scaled_sobel_h >= min_thresh) &
                    (scaled_sobel_h <= max_thresh)] = 1

    return binary_output_s, binary_output_l, binary_output_h


def edge_detection(img, min_thresh=10, max_thresh=255, dx=300, dy=200):
    equ1 = cv2.equalizeHist(img[:, :, 0])
    equ2 = cv2.equalizeHist(img[:, :, 1])
    equ3 = cv2.equalizeHist(img[:, :, 2])

    _img = np.dstack((equ1, equ2, equ3))
    blur = cv2.blur(_img, (5, 5))
    blur = cv2.blur(_img, (3, 3))
    edge_s, edge_l, edge_h = abs_sobel_(
        blur, min_thresh=min_thresh, max_thresh=max_thresh)

    output = np.zeros_like(edge_s)
    output[((edge_s == 1))] = 1
    width = img.shape[1]
    center = width // 2
    # print(center)
    contours = np.array([[center-dx, 700], [center+dx, 700], [center, 700-dy]])
    # print(contours)
    cv2.fillPoly(output, pts=[contours], color=(0))
    return output


def color_thersh_(img, thresh=0):
    white = white_threshold(img)
    yellow = yellow_threshold(img)
    img_ = yellow | white
    img_ = np.array(img_, dtype=np.uint8)

    output = np.zeros_like(img_)
    output[img_ > thresh] = 1
    return output


def white_threshold(rgbimg):
    HLS = cv2.cvtColor(rgbimg, cv2.COLOR_BGR2HLS)
    # HLS upper and lower limit for white colors
    lower = np.array([0, 200, 0])
    upper = np.array([255, 255, 255])
    masked_white = cv2.inRange(HLS, lower, upper)
    return masked_white


def yellow_threshold(rgbimg):
    HLS = cv2.cvtColor(rgbimg, cv2.COLOR_BGR2HLS)
    # HLS upper and lower limit for white colors
    lower = np.array([10, 0, 90])
    upper = np.array([40, 255, 255])
    masked_yellow = cv2.inRange(HLS, lower, upper)
    return masked_yellow


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    gradmag = np.sqrt(sobelx**2 + sobely**2)

    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi)):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output


def filter_image(img):
    mag_binary = mag_thresh(img, sobel_kernel=5, mag_thresh=(30, 255))
    dir_binary = dir_threshold(
        img, sobel_kernel=5, thresh=(45*np.pi/180, 75*np.pi/180))

    edge = edge_detection(img)
    binary_output = np.zeros_like(mag_binary)
    binary_output[(edge == 1)] = 1
    return binary_output


def warp(image):
    src = np.float32([[570, 470], [image.shape[1] - 573, 470],
                      [image.shape[1] - 150, image.shape[0]], [150, image.shape[0]]])
    dst = np.float32([[200, 0], [image.shape[1]-200, 0],
                      [image.shape[1]-200, image.shape[0]], [200, image.shape[0]]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    return warped


def inwarp(image):
    src = np.float32([[570, 460], [image.shape[1] - 573, 460],
                      [image.shape[1] - 150, image.shape[0]], [150, image.shape[0]]])
    dst = np.float32([[200, 0], [image.shape[1]-200, 0],
                      [image.shape[1]-200, image.shape[0]], [200, image.shape[0]]])
    M = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(
        image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    return warped


# def func(img):
#     return img


# for i in vid:
#     show_vid(i, filter_image)
#img = cv2.imread(test_images[2])
# cv2.imshow("Frame",img)
# # # edge = edge_canny(img)
# # # cv2.imshow("Frame",255*edge)
# cv2.waitKey(0)
# # cv2.destroyAllWindows()
