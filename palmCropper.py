from cvzone.HandTrackingModule import HandDetector


def cropPalm(img):
    detector = HandDetector(maxHands=1)
    hands, img = detector.findHands(img, draw=False)

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        # cropped_hand = img[y-20:y+20+h, x-20:x+20+w] #works for entire hand
        try:
            handType = hand["type"]
            if handType == "Right":
                cropped_hand = img[
                    y + int(h * 0.43) : y + int(h * 0.93),
                    x + int(w * 0.15) : x + int(w * 0.76),
                ]  # right hand ig
            else:
                cropped_hand = img[
                    y + int(h * 0.43) : y + int(h * 0.93),
                    x + int(w * 0.24) : x + int(w * 0.84),
                ]  # left hand ig
            return cropped_hand
        except:
            print("An exception occurred while cropping palm")
            return None

    return None
