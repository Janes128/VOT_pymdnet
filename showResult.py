# This function; print the picture of result.
import os
import cv2

def showResult(path):
    result_path = path
    img_list = os.listdir(result_path)
    img_list.sort()
    img_list = [os.path.join(result_path, x) for x in img_list]

    i = 0
    while cv2.waitKey(50) != ord('q') and i < len(img_list):
        print("Index #", i, ": ")
        image = cv2.imread(img_list[i])
        cv2.imshow("the result", image)
        i += 1
    print("Finish print")
