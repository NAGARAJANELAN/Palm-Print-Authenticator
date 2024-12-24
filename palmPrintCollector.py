import os
import cv2

from palmCropper import cropPalm

# make sure this directory is setted up
DATA_DIR = "./Palmprint/training/"

# each user's 10 sample palm image is stored
PALM_PRINT_COUNT = 8


def palmPrintCollector(name):
    if not os.path.exists(os.path.join(DATA_DIR)):
        os.makedirs(os.path.join(DATA_DIR))

    if not os.path.exists(os.path.join(DATA_DIR, name)):
        os.makedirs(os.path.join(DATA_DIR, name))

    cap = cv2.VideoCapture(0)

    print("Collecting palm print data for", name)

    while True:
        ret, frame = cap.read()
        cv2.putText(
            frame,
            'Press "r" when ready..',
            (100, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.imshow("Palmprint Collection", frame)
        if cv2.waitKey(25) == ord("r"):
            break

    counter = 0
    while counter < PALM_PRINT_COUNT:
        ret, frame = cap.read()
        cv2.imshow("Palmprint Collection in progress...", frame)
        frame = cropPalm(frame)
        if frame is None or len(frame) == 0:
            continue
        cv2.waitKey(500)
        cv2.imwrite(os.path.join(DATA_DIR, name, "pp{}.jpg".format(counter)), frame)
        counter += 1

    print("Palmprint Collected..")

    cap.release()
    cv2.destroyAllWindows()
