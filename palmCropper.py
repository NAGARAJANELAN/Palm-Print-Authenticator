from cvzone.HandTrackingModule import HandDetector


def cropPalm(img):
    detector = HandDetector(maxHands=1)
    hands, img = detector.findHands(img, draw=False)

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        # cropped_hand = img[y-20:y+20+h, x-20:x+20+w] #works for entire hand
        try:
            cropped_hand = img[
                y + int(h * 0.43) : y + int(h * 0.93),
                x + int(w / 6) : x + int(w * 0.73),
            ]  # right hand ig
            # cropped_hand = img[y-20:y+20+h, x-20:x+20+w]
            return cropped_hand
        except:
            print("An exception occurred")
            return None

    return None
