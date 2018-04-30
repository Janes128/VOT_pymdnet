import os
import sys
import time
from PIL import Image
from PIL import ImageGrab

def screenShot(pic_num):
    SaveDirectory= r'D:\pyMDNET\Temp'
    for i in range(int(pic_num)): 
        img = ImageGrab.grab();
        #saveas=os.path.join(SaveDirectory, 'ScreenShot_'+time.strftime("%Y-%m-%d_%H-%M-%S")+'.jpg') # Named by time
        if i < 10:
            saveas=os.path.join(SaveDirectory, '000' + str(i+2) +'.jpg')
        elif i < 100 and i > 9:
            saveas=os.path.join(SaveDirectory, '00' + str(i+2) +'.jpg')
        elif i < 1000 and i > 99:
            saveas=os.path.join(SaveDirectory, '00' + str(i+2) +'.jpg')

        img.save(saveas)
        print("Take " + str(i+1) + " of " + str(pic_num) + "...")
        time.sleep(0.5)
    sys.exit();

def firstShot():
    SaveDirectory= r'D:\pyMDNET\Temp'
    img = ImageGrab.grab();
    saveas=os.path.join(SaveDirectory, '0001' +'.jpg')
    img.save(saveas)
