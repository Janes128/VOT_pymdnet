import cv2, time, sys
import numpy as np
import ScreenShot as ss
from subprocess import call

global pic_num

# Open the init picture

# Parameters
drawing = False  # true if mouse pressed
mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1  # init target position


# mouse callback function
def draw_rect(event, x, y, flags, param):
    global ix, iy, drawing, mode
    # is Click
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        print("ix= ", ix, "iy= ", iy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 255), 2)
            tempx = min(ix, x)
            tempy = min(iy, y)
            tempw = abs(ix - x)
            temph = abs(iy - y)

            # Input the groundtruth
            initfile = open('D:\\pyMDNET\\Temp\\groundtruth_rect.txt', 'w')
            initfile_str = str(tempx) + ',' + str(tempy) + ',' + str(tempw) + ',' + str(temph)
            # for j in range(pic_num - 1):
            #     initfile_str += '\n0,0,0,0'
            initfile.write(initfile_str)
            #
            # time.sleep(1)
            # cv2.destroyAllWindows()
            # ss.screenShot(int(pic_num))
            sys.exit()


# Main function
if __name__ == "__main__":
    print("Welcome! Using pyMDNet VOT project...")
    print("Draw the first bbox: ")
    time.sleep(2)

    video_home = 'D:\\pyMDNET\\video/pig.mp4'
    video = cv2.VideoCapture(video_home)

    retval, image = video.read()
    cv2.namedWindow('Init_image')
    cv2.imshow("Init_image", image)
    cv2.setMouseCallback('Init_image', draw_rect)

    while (1):
        cv2.imshow('Init_image', image)
        k = cv2.waitKey(1) & 0xFF  # Waiting key and make sure that it's at least 8 bits
        if k == ord('m'):
            mode = not mode
        elif k == 27:  # Esc key to stop
            break

    # while cv2.waitKey(30) != ord('q'):
    #     retval, image = video.read()
    #     cv2.imshow("video", image)
    # video.release()
