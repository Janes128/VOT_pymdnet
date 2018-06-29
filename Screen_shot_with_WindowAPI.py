import time
import win32gui, win32ui, win32con, win32api


def window_capture(filename):
    hwnd = 0  # 窗口的編號，0號表示當前活躍窗口
    # 根據窗口句柄獲取窗口的設備上下文DC（Divice Context）
    hwndDC = win32gui.GetWindowDC(hwnd)
    # 根據窗口的DC獲取mfcDC
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    # mfcDC創建可兼容的DC
    saveDC = mfcDC.CreateCompatibleDC()
    # 創建bigmap準備保存圖片
    saveBitMap = win32ui.CreateBitmap()
    # 獲取監控器信息
    MoniterDev = win32api.EnumDisplayMonitors(None, None)
    w = MoniterDev[0][2][2]
    h = MoniterDev[0][2][3]
    # print w,h　　　#圖片大小
    # 為bitmap開闢空間
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
    # 高度saveDC，將截圖保存到saveBitmap中
    saveDC.SelectObject(saveBitMap)
    # 截取從左上角（0，0）長寬為（w，h）的圖片
    saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)
    saveBitMap.SaveBitmapFile(saveDC, filename)


#beg = time.time()
for i in range(10):
    window_capture("haha.jpg")
#end = time.time()
#print(end - beg)