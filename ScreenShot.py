import os
import sys
import time
from PIL import Image
from PIL import ImageGrab

SaveDirectory= r'D:\pyMDNET\Temp'

for i in range(5): 
    img = ImageGrab.grab();
    saveas=os.path.join(SaveDirectory, 'ScreenShot_'+time.strftime("%Y-%m-%d_%H-%M-%S")+'.jpg')
    img.save(saveas)
    time.sleep(5)