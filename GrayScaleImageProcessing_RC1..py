## GraS=None Image Processing using Python (Ver 2.2)
import math
import os.path
from cmath import cos, sin
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import *
from tkinter.simpledialog import *

## 함수부
def malloc2D(h, w, initValue =0) :
    memory = [[initValue for _ in range(w)] for _ in range(h)]
    return memory

def  openimage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    filename = askopenfilename(parent=window,
            filetypes=(("RAW 파일", "*.raw"), ("모든파일", "*.*")))
    # 중요! 입력 영상의 크기를 파악
    fSize = os.path.getsize(filename) # 파일 크기(Byte)
    inH = inW = int(math.sqrt(fSize)) # 256x256
    # 메모리 할당
    inImage = malloc2D(inH, inW)
    rfp = open(filename, 'rb')
    for i in range(inH):
        for k in range(inW):
            inImage[i][k] = ord(rfp.read(1))
    rfp.close()
    EqualImage()
def saveimage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    if outImage == None or len(outImage) == 0 : # 영상처리를 한적이 없다면
        return
    saveFp = asksaveasfile(parent=window, mode='wb', defaultextension="*.raw",
                          filetypes=(("RAW 파일", "*.raw"), ("All Files", "*.*")))
    import struct
    for i in range(outH) :
        for k in range(outW) :
            saveFp.write( struct.pack('B', outImage[i][k]) )
    saveFp.close()
    messagebox.showinfo("성공", saveFp.name + " 저장됨")

def draw() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, inpaper, outpaper
    if canvas != None :
        canvas.destroy()

    window.geometry('%dx%d' % (inW + outW + 10, inH))
    ## 캔버스/페이퍼 생성
    canvas = Canvas(window, height=outH, width=outW + inW)
    inpaper = PhotoImage(height=inH, width=inW)
    outpaper = PhotoImage(height=outH, width=outW)
    canvas.create_image((inW / 2, inH / 2), image=inpaper, state='normal')
    canvas.create_image((inW + outW / 2 + 10, inH / 2), image=outpaper, state='normal')

    rgbString=""

    for i in range(inH):
        tmpStringin = ""
        for k in range(inW):
            r = g = b = inImage[i][k]
            tmpStringin += "#%02x%02x%02x " % (r, g, b) # 제일 뒤에 공백
        rgbString += '{' + tmpStringin + '} ' # 제일 뒤에 공백
    inpaper.put(rgbString)

    rgbString = ""
    for i in range(outH):
        tmpStringout = ""
        for k in range(outW):
            r = g = b = outImage[i][k]
            tmpStringout += "#%02x%02x%02x " % (r, g, b) # 제일 뒤에 공백
        rgbString += '{' + tmpStringout + '} ' # 제일 뒤에 공백
    outpaper.put(rgbString)

    canvas.pack()


### 영상처리 함수 모음 ###
def EqualImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    ## ** 진짜 영상처리 알고리즘 ** ##
    for i in range(inH) :
        for k in range(inW) :
            outImage[i][k] = inImage[i][k]
    #################################
    draw()

def ReverseImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    ## ** 진짜 영상처리 알고리즘 ** ##
    for i in range(inH) :
        for k in range(inW) :
            outImage[i][k] = 255-inImage[i][k]
    #################################
    draw()
def BrightssImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    ## ** 진짜 영상처리 알고리즘 ** ##
    value = askinteger("밝기 이미지", "밝음 : + 어두움 : -", minvalue=-255, maxvalue=255)
    for i in range(inH) :
        for k in range(inW) :
            outImage[i][k] = inImage[i][k]+value
            if(outImage[i][k]>255):
                outImage[i][k]=255
            if (outImage[i][k] < 0):
                outImage[i][k]=0
    #################################
    draw()
def BrightmdImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    ## ** 진짜 영상처리 알고리즘 ** ##
    value = askfloat("밝기 이미지", "밝음 : * 어두움 : /", )
    for i in range(inH) :
        for k in range(inW) :
            outImage[i][k] = int(inImage[i][k]*value)
            if (outImage[i][k] > 255):
                outImage[i][k] = 255
            if (outImage[i][k] < 0):
                outImage[i][k] = 0
    #################################
    draw()
def BwImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    ## ** 진짜 영상처리 알고리즘 ** ##
    for i in range(inH) :
        for k in range(inW) :
            if(inImage[i][k]>127):
                outImage[i][k] = 255
            else :
                outImage[i][k] = 0
    #################################
    draw()
def BwavImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    ## ** 진짜 영상처리 알고리즘 ** ##
    hap=0
    for i in range(inH) :
        for k in range(inW) :
            hap+=inImage[i][k]
    avg=hap/(inH*inW)
    for i in range(inH) :
        for k in range(inW) :
            if(inImage[i][k]>avg):
                outImage[i][k] = 255
            else :
                outImage[i][k] = 0
    #################################
    draw()
def AndImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    value=askinteger("AND","입력 값", minvalue=-255, maxvalue=255)
    ## ** 진짜 영상처리 알고리즘 ** ##
    for i in range(inH) :
        for k in range(inW) :
            outImage[i][k]=inImage[i][k]&value
    #################################
    draw()
def OrImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    value=askinteger("OR","입력 값", minvalue=-255, maxvalue=255)
    ## ** 진짜 영상처리 알고리즘 ** ##
    for i in range(inH) :
        for k in range(inW) :
            outImage[i][k]=inImage[i][k]|value
    #################################
    draw()
def XorImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    value=askinteger("XOR","입력 값", minvalue=-255, maxvalue=255)
    ## ** 진짜 영상처리 알고리즘 ** ##
    for i in range(inH) :
        for k in range(inW) :
            outImage[i][k]=inImage[i][k]^value
    #################################
    draw()
def NotImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    ## ** 진짜 영상처리 알고리즘 ** ##
    for i in range(inH) :
        for k in range(inW) :
            outImage[i][k] = 256+~inImage[i][k]
    #################################
    draw()
def histostImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    ## ** 진짜 영상처리 알고리즘 ** ##
    low=inImage[0][0]
    high=inImage[0][0]
    for i in range(inH):
        for k in range(inW):
            if (inImage[i][k] < low):
                low = inImage[i][k]
            elif (inImage[i][k] > high):
                high = inImage[i][k]
        for i in range(inH):
            for k in range(inW):
                outImage[i][k] =int((inImage[i][k] - low) / (high - low) * 255.0)
    #################################
    draw()
def histoeiImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    ## ** 진짜 영상처리 알고리즘 ** ##
    low=inImage[0][0]
    high=inImage[0][0]
    for i in range(inH):
        for k in range(inW):
            if (inImage[i][k] < low):
                low = inImage[i][k]
            elif (inImage[i][k] > high):
                high = inImage[i][k]
    low += 50
    high -= 50
    for i in range(inH):
        for k in range(inW):
            px =(inImage[i][k] - low) / (high - low) * 255.
            if (px > 255):
                    px = 255
            if (px < 0):
                    px = 0
            outImage[i][k]=int(px)
    #################################
    draw()
def histoeqImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    ## ** 진짜 영상처리 알고리즘 ** ##
    hist=[0 for _ in range(256)]
    for i in range(inH):
        for k in range(inW):
            hist[inImage[i][k]]+=1
    sumhist = [0 for _ in range(256)]
    sumhist[0]=hist[0]
    for i in range(1, 256, 1):
        sumhist[i]=hist[i]+sumhist[i-1]
    normalhist = [0 for _ in range(256)]
    for i in range(256):
        normalhist[i] = sumhist[i] * (1.0 / (inW * inH)) * 255.0
    for i in range(inH):
        for k in range(inW):
            outImage[i][k] = int(normalhist[inImage[i][k]])
    draw()
def GammaImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    ## ** 진짜 영상처리 알고리즘 ** ##
    value = askinteger("감마 보정","입력 값")
    for i in range(inH):
        for k in range(inW):
            outImage[i][k] =255 * pow((inImage[i][k] / 255), value)
            if(inImage[i][k]>127):
                outImage[i][k] = 255
            else :
                outImage[i][k] = 0
    #################################
    draw()
def StressImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    ## ** 진짜 영상처리 알고리즘 ** ##
    range1=askinteger("범위 강조","처음 값")
    range2=askinteger("범위 강조","끝 값")
    for i in range(inH):
        for k in range(inW):
            if(range1 <= inImage[i][k]) & (inImage[i][k] <= range2):
                outImage[i][k] = 255
            else:
                outImage[i][k] = inImage[i][k]
    #################################
    draw()
## 화소 영역처리 #############################################################
def embImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    ## ** 진짜 영상처리 알고리즘 ** ##

    tmpIn = malloc2D(inH + 2, inW + 2)
    tmpOut = malloc2D(outH, outW)

    MASK = [[-1, 0, 0], [0, 0, 0], [0, 0, 1]]
    for i in range(inH + 2) :
        for k in range(inW + 2) :
            tmpIn[i][k] = 0.


    for i in range(inH):
        for k in range(inW):
            tmpIn[i + 1][k + 1] = inImage[i][k]
    S=0.0
    for i in range(inH):
        for k in range(inW):
            S = 0.
            for m in range(3):
                for n in range(3):
                    S += tmpIn[i + m][k + n] * MASK[m][n]
            tmpOut[i][k] = S
    for i in range(inH):
        for k in range(inW):
            tmpOut[i][k] += 127.

    for i in range(outH):
        for k in range(outW):
            v = tmpOut[i][k]
            if (v > 255):
                v = 255.
            if (v < 0):
                v = 0.
            outImage[i][k] = int(v)
    draw()
def blurImage():
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    ## ** 진짜 영상처리 알고리즘 ** ##

    tmpIn = malloc2D(inH + 2, inW + 2)
    tmpOut = malloc2D(outH, outW)

    MASK =[ [  1. / 9,  1. / 9,  1. / 9],
		[  1. / 9,  1. / 9,  1. / 9],
		[  1. / 9,  1. / 9,  1. / 9]]
    for i in range(inH + 2):
        for k in range(inW + 2):
            tmpIn[i][k] = 0.

    for i in range(inH):
        for k in range(inW):
            tmpIn[i + 1][k + 1] = inImage[i][k]
    S = 0.0
    for i in range(inH):
        for k in range(inW):
            S = 0.
            for m in range(3):
                for n in range(3):
                    S += tmpIn[i + m][k + n] * MASK[m][n]
            tmpOut[i][k] = S
    for i in range(inH):
        for k in range(inW):
            tmpOut[i][k] += 127.

    for i in range(outH):
        for k in range(outW):
            v = tmpOut[i][k]
            if (v > 255):
                v = 255.
            if (v < 0):
                v = 0.
            outImage[i][k] = int(v)
    draw()
def guasImage():
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    ## ** 진짜 영상처리 알고리즘 ** ##

    tmpIn = malloc2D(inH + 2, inW + 2)
    tmpOut = malloc2D(outH, outW)

    MASK = [ [  1. / 16,  1. / 8,  1. / 16],
		[  1. / 8,  1. / 4,  1. / 8],
		[  1. / 16,  1. / 8,  1. / 16]]
    for i in range(inH + 2):
        for k in range(inW + 2):
            tmpIn[i][k] = 0.

    for i in range(inH):
        for k in range(inW):
            tmpIn[i + 1][k + 1] = inImage[i][k]
    S = 0.0
    for i in range(inH):
        for k in range(inW):
            S = 0.
            for m in range(3):
                for n in range(3):
                    S += tmpIn[i + m][k + n] * MASK[m][n]
            tmpOut[i][k] = S
    for i in range(inH):
        for k in range(inW):
            tmpOut[i][k] += 127.

    for i in range(outH):
        for k in range(outW):
            v = tmpOut[i][k]
            if (v > 255):
                v = 255.
            if (v < 0):
                v = 0.
            outImage[i][k] = int(v)
    draw()
def sharpImage():
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    ## ** 진짜 영상처리 알고리즘 ** ##

    tmpIn = malloc2D(inH + 2, inW + 2)
    tmpOut = malloc2D(outH, outW)

    MASK =  [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]
    for i in range(inH + 2):
        for k in range(inW + 2):
            tmpIn[i][k] = 0.

    for i in range(inH):
        for k in range(inW):
            tmpIn[i + 1][k + 1] = inImage[i][k]
    S = 0.0
    for i in range(inH):
        for k in range(inW):
            S = 0.
            for m in range(3):
                for n in range(3):
                    S += tmpIn[i + m][k + n] * MASK[m][n]
            tmpOut[i][k] = S
    for i in range(inH):
        for k in range(inW):
            tmpOut[i][k] += 127.

    for i in range(outH):
        for k in range(outW):
            v = tmpOut[i][k]
            if (v > 255):
                v = 255.
            if (v < 0):
                v = 0.
            outImage[i][k] = int(v)
    draw()
def hpfImage():
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    ## ** 진짜 영상처리 알고리즘 ** ##

    tmpIn = malloc2D(inH + 2, inW + 2)
    tmpOut = malloc2D(outH, outW)

    MASK = [[-1. / 9, -1. / 9, -1. / 9],
            [-1. / 9, 8. / 9, -1. / 9],
            [-1. / 9, -1. / 9, -1. / 9]]
    for i in range(inH + 2):
        for k in range(inW + 2):
            tmpIn[i][k] = 0.

    for i in range(inH):
        for k in range(inW):
            tmpIn[i + 1][k + 1] = inImage[i][k]
    S = 0.0
    for i in range(inH):
        for k in range(inW):
            S = 0.
            for m in range(3):
                for n in range(3):
                    S += tmpIn[i + m][k + n] * MASK[m][n]
            tmpOut[i][k] = S
    for i in range(inH):
        for k in range(inW):
            tmpOut[i][k] += 127.

    for i in range(outH):
        for k in range(outW):
            v = tmpOut[i][k]
            if (v > 255):
                v = 255.
            if (v < 0):
                v = 0.
            outImage[i][k] = int(v)
    draw()
## 기하학 처리############################################################################
def LrImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    ## ** 진짜 영상처리 알고리즘 ** ##
    for i in range(inH):
        for k in range(inW):
            outImage[i][(inW - 1) - k] = inImage[i][k]
    #################################
    draw()
def UdImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    ## ** 진짜 영상처리 알고리즘 ** ##
    for i in range(inH):
        for k in range(inW):
            outImage[(inH - 1) -i][ k] = inImage[i][k]
    #################################
    draw()
def LrudImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, canvas, paper
    # 중요! 출력영상의 크기를 결정 --> 알고리즘에 의존
    outH = inH
    outW = inW
    # 메모리 할당
    outImage = malloc2D(outH, outW)
    ## ** 진짜 영상처리 알고리즘 ** ##
    for i in range(inH):
        for k in range(inW):
            outImage[(inH - 1) -i][(inW - 1) -k] = inImage[i][k]
    #################################
    draw()
def RotateImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, paper, canvas

    angle = askinteger("이미지 회전","회전할 각도 입력")
    radian = math.radians(angle)

    outH = int(inH * abs(math.cos(radian)) + inW * abs(math.cos(math.pi / 2 - radian)))
    outW = int(inH * abs(math.cos(math.pi / 2 - radian)) + inW * abs(math.cos(radian)))

    outImage = malloc2D(outH, outW)

    cx = outH / 2
    cy = outW / 2

    cx1 = inH / 2
    cy1 = inW / 2

    for xd in range (outH):
        for yd in range(outW):
            xs = int(math.cos(radian) * (xd - cx) + math.sin(radian) * (yd - cy) + cx1)
            ys = int(-math.sin(radian)* (xd - cx) + math.cos(radian) * (yd - cy) + cy1)

            if (0 <= xs and xs < inH) and (0 <= ys and ys < inW):
                outImage[xd][yd] = inImage[xs][ys]
    #################################
    draw()
def MoveImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, paper, canvas

    outH = inH
    outW = inW

    outImage = malloc2D(outH, outW)

    movx = askinteger("이동", "x축 이동할 값(-" + str(inH) + "~" + str(inH) + ") 입력", minvalue=-inH, maxvalue=inH)
    movy = askinteger("이동", "y축 이동할 값(-" + str(inW) + "~" + str(inW) + ") 입력", minvalue=-inW, maxvalue=inW)

    for i in range(outH):
        for k in range(outW):
            if i + movy < outH and k + movx < outW and i + movy > 0 and k + movx > 0:
                outImage[i + movy][k + movx] = inImage[i][k]
#######################
    draw()

def ZiImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, paper, canvas
    scale = askinteger("이미지 확대", "확대할 배율(2~5) 입력하세요", minvalue=2, maxvalue=5)
    outH = inH * scale
    outW = inW * scale
    outImage = malloc2D(outH, outW)

    for i in range(outH):
        for k in range(outW):
            outImage[i][k] = inImage[int(i / scale)][int(k / scale)]
##########################################
    draw()
def ZoImage() :
    global inImage, outImage, inH, inW, outH, outW
    global window, paper, canvas
    scale = askinteger("이미지 축소", "축소할 배율(2~5) 입력하세요", minvalue=2, maxvalue=5)
    outH = int(inH / scale)
    outW = int(inW / scale)
    outImage = malloc2D(outH, outW)

    for i in range(inH):
        for k in range(inW):
            outImage[int(i / scale)][int(k / scale)] = inImage[i][k]
#######################################
    draw()
## 전역변수부
inImage, outImage = [], []  # unsinged char ** m_inImage, ** m_outImage
inH, inW, outH, outW = [0] * 4
window, canvas, paper = None, None, None

## 메인코드부
window = Tk() # 벽
window.title("Gray영상처리 Ver2.2")
window.geometry('500x500')

mainMenu = Menu(window) # 메뉴의 틀
window.config(menu=mainMenu)

fileMenu = Menu(mainMenu) # 상위 메뉴(파일)
mainMenu.add_cascade(label='파일', menu=fileMenu)
fileMenu.add_command(label='열기', command=openimage)
fileMenu.add_command(label='저장', command=saveimage)
fileMenu.add_separator()
fileMenu.add_command(label='종료')

image1Menu = Menu(mainMenu) # 상위 메뉴(파일)
mainMenu.add_cascade(label='화소점처리', menu=image1Menu)
image1Menu.add_command(label='동일 이미지', command=EqualImage)
image1Menu.add_command(label='밝기 이미지(+,-)', command=BrightssImage)
image1Menu.add_command(label='밝기 이미지(*,/)', command=BrightmdImage)
image1Menu.add_command(label='반전 이미지', command=ReverseImage)
image1Menu.add_command(label='흑백 이미지', command=BwImage)
image1Menu.add_command(label='흑백(평균) 이미지', command=BwavImage)
image1Menu.add_separator()
image1Menu.add_command(label='AND', command=AndImage)
image1Menu.add_command(label='OR', command=OrImage)
image1Menu.add_command(label='XOR', command=XorImage)
image1Menu.add_command(label='NOT', command=NotImage)
image1Menu.add_separator()
image1Menu.add_command(label='히스토그램(스트래칭)', command=histostImage)
image1Menu.add_command(label='히스토그램(엔드-인 탐색)', command=histoeiImage)
image1Menu.add_command(label='히스토그램(평활화)', command=histoeqImage)
image1Menu.add_command(label='감마 보정', command=GammaImage)
image1Menu.add_command(label='범위 강조', command=StressImage)

image2Menu = Menu(mainMenu) # 상위 메뉴(파일)
mainMenu.add_cascade(label='화소영역처리', menu=image2Menu)
image2Menu.add_command(label='엠보싱', command=embImage)
image2Menu.add_command(label='블러링', command=blurImage)
image2Menu.add_command(label='가우시안', command=guasImage)
image2Menu.add_command(label='샤프닝', command=sharpImage)
image2Menu.add_command(label='고주파통과필터', command=hpfImage)


image3Menu = Menu(mainMenu) # 상위 메뉴(파일)
mainMenu.add_cascade(label='기하학 처리', menu=image3Menu)
image3Menu.add_command(label='좌우 이미지', command=LrImage)
image3Menu.add_command(label='상하 이미지', command=UdImage)
image3Menu.add_command(label='좌우+상하 이미지', command=LrudImage)
image3Menu.add_command(label='자유 회전', command=RotateImage)
image3Menu.add_separator()
image3Menu.add_command(label='이동', command=MoveImage)
image3Menu.add_command(label='확대', command=ZiImage)
image3Menu.add_command(label='축소', command=ZoImage)

window.mainloop()
