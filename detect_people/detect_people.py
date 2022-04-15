# import the necessary packages
import numpy as np
from datetime import datetime, timedelta
import cv2


class DetectPeople:
    def __init__(self):
        # initialize the HOG descriptor/person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        cv2.startWindowThread()

        # open webcam video stream
        self.cap = cv2.VideoCapture(0)

        # the output will be written to output.avi
        self.out = cv2.VideoWriter(
            'output.avi',
            cv2.VideoWriter_fourcc(*'MJPG'),
            15.,
            (640, 480))

    def detect_people(self):
        # run loop for 15 seconds
        # start_time
        start_time = datetime.now()
        len_boxes = []

        while(True):
            # Capture frame-by-frame
            ret, frame = self.cap.read()

            # resizing for faster detection
            frame = cv2.resize(frame, (640, 480))
            # turn to greyscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # apply threshold. all pixels with a level larger than 80 are shown in white. the others are shown in black:
            ret, frame = cv2.threshold(frame, 80, 255, cv2.THRESH_BINARY)

            # detect people in the image
            # returns the bounding boxes for the detected objects
            boxes, weights = self.hog.detectMultiScale(frame, winStride=(8, 8))

            boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
            print(len(boxes))
            len_boxes.append(len(boxes))

            for (xA, yA, xB, yB) in boxes:
                # display the detected boxes in the colour picture
                cv2.rectangle(frame, (xA, yA), (xB, yB),
                              (0, 255, 0), 2)

            # Write the output video
            self.out.write(frame.astype('uint8'))
            # Display the resulting frame
            cv2.imshow('frame', frame)
            # current time
            current_time = datetime.now()
            # if current_time - start_time > 15 seconds, break
            if current_time - start_time > timedelta(seconds=15):
                break

        total_values = len(len_boxes)
        zero_values = len([i for i in len_boxes if i == 0])
        print(f"Total values: {total_values}")
        print(f"Zero values: {zero_values}")
        if(zero_values / total_values > 0.5):
            print("No people detected")
            return False
        else:
            print("People detected")
            return True

    def __del__(self):
        # When everything done, release the capture
        self.cap.release()
        # and release the output
        self.out.release()
        # finally, close the window
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    detect_people = DetectPeople()
    detect_people.detect_people()
