import numpy as np
from datetime import datetime, timedelta
import cv2

class HumanDetector:
    """Detects humans in the camera feed."""
    def __init__(self, accuracy_factor=0.5, num_iter=10, write_video=False, seconds = 10, display_box = False):
        """Initializes the human detector."""
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        cv2.startWindowThread()
        self.num_iter = num_iter
        self.accuracy_factor = accuracy_factor
        self.seconds = seconds
        self.display_box = display_box
        if write_video:
            self.out = cv2.VideoWriter(
                'output.avi',
                cv2.VideoWriter_fourcc(*'MJPG'),
                15.,
                (640,480))
        else:
            self.out = None


    def __del__(self):
        """Closes the camera feed."""
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def detect_people_iterations(self):
        """Detects people in the camera feed for a number of iterations."""
        self.cap = cv2.VideoCapture(0)
        num_people_detected = 0
        for i in range(self.num_iter):
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            boxes, weights = self.hog.detectMultiScale(frame, winStride=(8,8))
            if len(boxes) > 0:
                num_people_detected += 1
            if self.display_box:
                boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])  
                for (xA, yA, xB, yB) in boxes:
                    # display the detected boxes in the colour picture
                    cv2.rectangle(frame, (xA, yA), (xB, yB),
                                    (0, 255, 0), 2)
                cv2.imshow('frame',frame)

            if self.out is not None:
                self.out.write(frame.astype('uint8'))
        
        self.cap.release()
        if num_people_detected >= self.accuracy_factor * self.num_iter:
            return True
        return False

    def detect_people_timedelta(self):
        """Detects people in the camera feed for a number of seconds."""
        start_time = datetime.now()
        num_people_detected = 0
        total_iterations = 0
        self.cap = cv2.VideoCapture(0)
        print("Starting to detect people")
        while datetime.now() < start_time + timedelta(seconds=self.seconds):
            total_iterations += 1
            ret, frame = self.cap.read()
            if frame is None:
                continue
            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            boxes, weights = self.hog.detectMultiScale(frame, winStride=(8,8))
            if len(boxes) > 0:
                num_people_detected += 1

            if self.display_box:
                boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])  
                for (xA, yA, xB, yB) in boxes:
                    # display the detected boxes in the colour picture
                    cv2.rectangle(frame, (xA, yA), (xB, yB),
                                    (0, 255, 0), 2)
                cv2.imshow('frame',frame)
                cv2.waitKey(1)

            if self.out is not None:
                self.out.write(frame.astype('uint8'))
        print("Finished detecting people")
        self.cap.release()
        if (num_people_detected*1.0) / (total_iterations*1.0) >= self.accuracy_factor:
            return True
        print(num_people_detected, total_iterations, self.accuracy_factor)
        return False

if __name__ == "__main__":
    dp = HumanDetector(display_box=True, write_video=True, seconds=10, accuracy_factor=0.1)
    # print start time
    print("Start time: ", datetime.now())
    print(dp.detect_people_timedelta())
    del dp
    # print end time
    print("End time: ", datetime.now())
