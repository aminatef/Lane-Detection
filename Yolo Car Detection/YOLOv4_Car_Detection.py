from tkinter import Label
import numpy as np
import time
import cv2
import sys
# import matplotlib.pyplot as plt
# from moviepy.editor import VideoFileClip


def show_vid(name, func):
    capture = cv2.VideoCapture(name)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)+0.5)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)+0.5)
    frame_size = (width, height)
    output = cv2.VideoWriter('OUT_YOLOV4.mp4', cv2.VideoWriter_fourcc(
        *'mp4v'), 25, frame_size, isColor=True)
    if (capture.isOpened() == False):
        print("Error opening video  file")
    while(capture.isOpened()):
        return_, frame = capture.read()
        if return_:
            # cv2.imshow("Frame",255*func(frame))
            # cv2.imshow("Frame",frame)
            output.write(func(frame))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    capture.release()
    cv2.destroyAllWindows()


class yolo_model():

    def __init__(self, config, weights, labelPath):
        self.LABELS = open(labelPath).read().splitlines()
        net = cv2.dnn.readNetFromDarknet(config, weights)
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(scale=1 / 255, size=(192, 192), swapRB=True)

    def predict(self, image):
        classIds, scores, boxes = self.model.detect(
            image, confThreshold=0.6, nmsThreshold=0.4)

        for i, (classId, score, box) in enumerate(zip(classIds, scores, boxes)):
            cv2.rectangle(
                image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

            text = '%s%s: %.2f' % (self.LABELS[classId], i, score)
            cv2.putText(image, text, (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return image


def main(cfg_path, weights_path, Labels_path, video_path):
    yolo = yolo_model(cfg_path, weights_path, Labels_path)
    start = time.time()
    show_vid(video_path, yolo.predict)
    print(time.time()-start)


if __name__ == '__main__':
    cfg_path = sys.argv[1]
    weights_path = sys.argv[2]
    Labels_path = sys.argv[3]
    video_path = sys.argv[4]
    main(cfg_path, weights_path, Labels_path, video_path)
