import os
import cv2
from palmCropper import cropPalm
from palmMatchScorer import palmMatchScore

DATA_DIR = "./Palmprint/training/"
PALM_PRINT_COUNT = 5
THRESHOLD = 56
STORED_PALM_COUNT = 8


def palmPrintAuthenticate(username):
    if not os.path.exists(os.path.join(DATA_DIR, username)):
        print("User doesn't exist..")
        return False  # user not registered

    cap = cv2.VideoCapture(0)

    print("Collecting palm print data for", username)

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
        cv2.imshow("Palmprint Authentication", frame)
        if cv2.waitKey(25) == ord("r"):
            break

    # cap.release()
    cv2.destroyAllWindows()  # clears first window
    # cap = cv2.VideoCapture(0)

    counter = 0
    while counter < PALM_PRINT_COUNT:
        ret, frame = cap.read()
        cv2.imshow("Palmprint Authentication in progress...", frame)
        frame = cropPalm(frame)
        if frame is None or len(frame) == 0:
            continue

        # compare with images got during login
        total_score = 0
        for i in range(0, STORED_PALM_COUNT):
            total_score += palmMatchScore(
                frame, os.path.join(DATA_DIR, username + "/pp{}.jpg".format(i)), 0.8
            )

        print("Score : ", total_score)
        if total_score > THRESHOLD:
            print("Valid user")
            return True

        cv2.waitKey(200)
        counter += 1

    print("Palmprint Collected..")

    cap.release()
    cv2.destroyAllWindows()

    print("Invalid user")

    return False
