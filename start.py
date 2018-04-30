import cv2,time,sys
import numpy as np
import ScreenShot as ss
from subprocess import call            
global pic_num

# Opne the init picture

# Parameters
drawing = False     # true if mouse pressed
mode = True         # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1     # init target position

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
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 255),2)
            tempx = min(ix, x)
            tempy = min(iy, y)
            tempw = abs(ix-x)
            temph = abs(iy-y)
            #initfile = open('C:\\Users\\user\\py-mdnet\\dataset\\OBT\\groundtruth_rect.txt','w')
            # Input the groundtruth
            initfile = open('D:\\pyMDNET\\Temp\\groundtruth_rect.txt','w')
            initfile_str = str(tempx)+','+str(tempy)+','+str(tempw)+','+str(temph)
            for j in range(pic_num-1):
                initfile_str += '\n0,0,0,0'
            initfile.write(initfile_str)
            
            time.sleep(1)
            #call(["D:\\Program files\\MATLAB\\R2017b\\bin\\matlab.exe","-nodisplay","-nosplash","-nodesktop","-r","run('C:\\Users\\user\\Desktop\\ECO_project\\ECOcode\\ECO-master\\demo_ECO_HC.m')",])
            cv2.destroyAllWindows()
            ss.screenShot(int(pic_num))
            sys.exit()

# Main function
if __name__ == "__main__":
    print("Welcome! Using pyMDNet VOT project...")
    pic_num = int(input("Please input: How many picture do you want to track? input: "))
    if pic_num > 9999:
        alert("It's out of range...")
        sys.exit();
    input("Please Press Enter to start...")

    couter = 3;
    for i in range(3):
        print("After " + str(couter) + " seconds to take first shot")
        couter -= 1;
        time.sleep(1)

    ss.firstShot();

    img = cv2.imread('D:\\pyMDNET\\Temp\\0001.jpg')
    cv2.namedWindow('Init_image')
    cv2.imshow('Init_image',img)
    cv2.setMouseCallback('Init_image', draw_rect)
    
    while(1):
            #s,img = cam.read()
            #img=cv2.flip(img,1)
            #cv2.imwrite("room.jpg",img)
        cv2.imshow('Init_image',img)
        k = cv2.waitKey(1) & 0xFF           # Waiting key and make sure that it's at least 8 bits          
        if k == ord('m'):
            mode = not mode
        elif k == 27:                       # Esc key to stop
            break 

