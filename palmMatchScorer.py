import cv2


def get_train_and_test_img_features(path):
    if type(path) == type("hell"):
        img = cv2.imread(path)  # except this line remove other lines from if else block
    else:
        img = path
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalize = cv2.equalizeHist(gray)  # step1.预处理图片，灰度均衡化
    kp_query, des_query = get_sift_features(
        equalize
    )  # step2.获取SIFT算法生成的关键点kp和描述符des(特征描述向量)
    return des_query


def get_sift_features(img, dect_type="sift"):
    if dect_type == "sift":
        sift = cv2.xfeatures2d.SIFT_create()
    elif dect_type == "surf":
        sift = cv2.xfeatures2d.SURF_create()
    kp, des = sift.detectAndCompute(img, None)  # kp为关键点，des为描述符
    return kp, des


def sift_detect_match_num(des_query, des_train, ratio=0.75):
    # step3.使用KNN计算查询图像与训练图像之间匹配的点数目,采用k(k=2)近邻匹配，最近的点距离与次近点距离之比小于阈值ratio就认为是成功匹配。
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_query, des_train, k=2)
    match_num = 0
    for first, second in matches:
        if first.distance < ratio * second.distance:
            match_num = match_num + 1
    return match_num


def palmMatchScore(img1, img2, ratio):  # <-----------------Utility function
    dq = get_train_and_test_img_features(img1)
    dt = get_train_and_test_img_features(img2)
    return sift_detect_match_num(dq, dt, ratio)
