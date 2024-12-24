import cv2


def get_train_and_test_img_features(path):
    if type(path) == type("hell"):
        img = cv2.imread(path)
    else:
        img = path
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalize = cv2.equalizeHist(gray)
    kp_query, des_query = get_sift_features(
        equalize
    )
    return des_query


def get_sift_features(img, dect_type="sift"):
    if dect_type == "sift":
        sift = cv2.xfeatures2d.SIFT_create()
    elif dect_type == "surf":
        sift = cv2.xfeatures2d.SURF_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


def sift_detect_match_num(des_query, des_train, ratio=0.75):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_query, des_train, k=2)
    match_num = 0
    for first, second in matches:
        if first.distance < ratio * second.distance:
            match_num = match_num + 1
    return match_num


def palmMatchScore(img1, img2, ratio):
    dq = get_train_and_test_img_features(img1)
    dt = get_train_and_test_img_features(img2)
    return sift_detect_match_num(dq, dt, ratio)
