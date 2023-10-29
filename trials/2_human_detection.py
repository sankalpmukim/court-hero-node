# import the necessary packages
import numpy as np
import cv2

print("[INFO] Loading HOG descriptor/person detector...")
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()
print("[INFO] Starting window thread...")

# open webcam video stream
print("[INFO] Opening webcam video stream...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Unable to open the webcam. Exiting...")
    exit()

# the output will be written to output.avi
print("[INFO] Setting up video writer for 'output.avi'...")
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (640, 480))

print("[INFO] Starting person detection...")
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Failed to grab frame. Continuing...")
        continue

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
    if len(boxes) > 0:
        print(f"[DETECTION] Detected {len(boxes)} person(s)...")

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for xA, yA, xB, yB in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # Write the output video
    out.write(frame.astype("uint8"))
    # Display the resulting frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("[INFO] 'q' key pressed. Exiting loop...")
        break

# When everything done, release the capture
print("[INFO] Releasing video capture...")
cap.release()
# and release the output
print("[INFO] Releasing video writer...")
out.release()
# finally, close the window
print("[INFO] Closing windows and cleaning up...")
cv2.destroyAllWindows()
cv2.waitKey(1)
print("[INFO] Done.")
