import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import sys


def show_vid(name, func):
    capture = cv2.VideoCapture(name)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)+0.5)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)+0.5)
    frame_size = (width, height)
    output = cv2.VideoWriter('out_edge.mp4', cv2.VideoWriter_fourcc(
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

    def __init__(self, config, weights, labelPath, thresh=0.8):
        self.CONFIDENCE_THRESHOLD = thresh
        self.LABELS = open(labelPath).read().strip().split("\n")
        np.random.seed(4)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
                                        dtype="uint8")
        self.net = cv2.dnn.readNetFromDarknet(config, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def predict(self, image):
        (H, W) = image.shape[:2]

        # determine only the *output* layer names that we need from YOLO
        ln = self.net.getLayerNames()
        ln = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (320, 320),
                                     swapRB=False, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(ln)
        end = time.time()

        # print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs

        for output in layerOutputs:
            # loop over each of the detections

            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.CONFIDENCE_THRESHOLD:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESHOLD,
                                0.3)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            predictions = dict()
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # print((x,y,x+w,y+h),f'{i}')

                color = [int(c) for c in self.COLORS[classIDs[i]]]
                # if not classIDs[i] in predictions.keys():
                # 	predictions[classIDs[i]]=list()
                # predictions[classIDs[i]].append((x, y, w, h,confidences[i]))

                # draw BB on image
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(
                    self.LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
                # cv2.putText(image, f'{i}', (int(x+w/2), y+h + 5), cv2.FONT_HERSHEY_SIMPLEX,
                # 	0.5, color, 2)
        return image


def main(cfg_path, weights_path, dataset_path):
    yolo = yolo_model(cfg_path, weights_path, dataset_path)
    # pred_img = yolo.predict(cv2.imread('img112.jpg'))
    # start = time.time()
    show_vid("project_video.mp4", yolo.predict)
    # print(time.time()-start)

    # plt.imshow(pred_img)
    # plt.show()


if __name__ == '__main__':
    cfg_path = sys.argv[1]
    weights_path = sys.argv[2]
    dataset_path = sys.argv[3]
    main(cfg_path, weights_path, dataset_path)
