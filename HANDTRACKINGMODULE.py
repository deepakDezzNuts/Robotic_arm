import math
import cv2 # cv library
import mediapipe as mp
import time
class HandDetector:
    def __init__(self, mode=False, maxhands=2, modelComplexity=1, detectioncon=0.5, trackingcon=0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.modelComplex = modelComplexity
        self.detectioncon = detectioncon
        self.trackingcon = trackingcon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxhands, self.modelComplex, self.detectioncon, self.trackingcon)
        self.mpdraw = mp.solutions.drawing_utils
        self.tipids = [4, 8, 12, 16, 20]

    def findhands(self, img, draw=True):
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRgb)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw: self.mpdraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
        return img
    def findPosition(self, img, handNo=0, draw=True):
        self.lmlst = []
        if self.results.multi_hand_landmarks:
            myhands = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myhands.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmlst.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0 , 0 ,255), cv2.FILLED)
        return self.lmlst
    def FindHandsUp(self):
        fingers = []
        if self.lmlst[self.tipids[0]][1] > self.lmlst[self.tipids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if self.lmlst[self.tipids[id]][2] < self.lmlst[self.tipids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmlst[p1][1:]
        x2, y2 = self.lmlst[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0
    cTime = 0
    vc = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        i, img = vc.read()
        img = detector.findhands(img)
        lmlst = detector.findposition(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 100), cv2.FONT_ITALIC, 3, (255, 0, 255), 4)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()