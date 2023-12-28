import math
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import cvzone
###############################################
from pyfirmata import Arduino, SERVO, util
from time import sleep

port = 'COM7'
pin = 10
pin1 = 11
pin2 = 12

board = Arduino(port)

board.digital[pin].mode = SERVO
board.digital[pin1].mode = SERVO
board.digital[pin2].mode = SERVO


def rotateservo(pin,angle):
    print("z")
    board.digital[pin].write(angle)
    sleep(0.0015)
   # board.digital[pin].write(angle)
    #sleep(0.015)
    #board.digital[pin2].write(angle2)
    #sleep(0.015)
def rotateservo1(pin1, angle1):
    print("x")
    board.digital[pin1].write(angle1)
    sleep(0.0015)
def rotateservo2(pin2, angle2):
    print("y")
    board.digital[pin2].write(angle2)
    sleep(0.015)
#############################################


########################
wCam, hCam = 1280, 720  #
frameR = 50           #
smooth = 2             #
clocx,clocy=0,0      #
plocx,plocy=0,0  #
########################
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
detector = HandDetector(detectionCon=0.8,maxHands=4)

##################################################
x=[300,245,200,170,145,130,112,103,93,87,80,75,70,67,62,59,57]
y=[20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
coff=np.polyfit(x,y,2)
#################################################
while True:
    success, img = cap.read()
    hands,img=detector.findHands(img)
    if hands:
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        lmList=hands[0]['lmList']
        x,y,w,h=hands[0]['bbox']
        # print(lmList)
        x1,y1,z1=lmList[5]
        x2,y2,z2=lmList[17]
        x3, y3,z3= lmList[9]
        x4 = np.interp(x3, (frameR, wCam - frameR), (0, 180))
        y4 = np.interp(y3, (frameR, hCam - frameR), (0, 180))
        # clocx = plocx + (x4 - plocx) / smooth
        # clocy = plocy + (y4 - plocy) / smooth
        distance=math.sqrt(((y2-y1)**2+(x2-x1)**2))
        A,B,C=coff
        distanceCM=A*distance**2 +B*distance+C
        #print(distanceCM,distance)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),3)
        cv2.circle(img, (x3, y3), 50, (255, 0, 255), cv2.FILLED)
        print(x4,y4,distanceCM,type(distanceCM))
        z4=distanceCM
        board.digital[pin2].write(int(z4))
        sleep(0.0015)
        board.digital[pin].write(int(x4))
        sleep(0.0015)
        #board.digital[pin2].write(int(y4))
        #sleep(0.01)
        #rotateservo(pin,int(z4))
        #rotateservo1(pin1, int(x4))
        #rotateservo2(pin2, int(y4))
        cvzone.putTextRect(img,f'{int(distanceCM)} cm',(x,y))
    # start=(int)(1280/2)
    # end=(int)(720/2)
    # cv2.line(img,(start,0),(start,720),(255, 0, 255),5)
    # cv2.line(img, (0, end), (1280, end), (255, 0, 255), 5)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

