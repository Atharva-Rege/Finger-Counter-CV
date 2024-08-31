#To use this module -> import HandTrackingModule as htm


import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, tracking_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.tracking_con = tracking_con

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.max_hands,
                                        model_complexity=1,
                                        min_detection_confidence=self.detection_con,
                                        min_tracking_confidence=self.tracking_con)

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # As hands uses only RGB
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLandmks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLandmks, self.mpHands.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, hand_no=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            for handLandmks in self.results.multi_hand_landmarks:
                my_hand = self.results.multi_hand_landmarks[hand_no]
                for id, lm in enumerate(my_hand.landmark):
                    # print(id,lm)
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id, cx, cy)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(frame, (cx, cy), 5, (255, 255, 0), cv2.FILLED)
        return lmList


def main():
    pTime = 0.
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        ret, frame = cap.read()
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame)
        if len(lmList) != 0:
            #print(lmList[4])
            pass
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, "FPS: " + str(int(fps)), (10, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()

'''
Example Project:-

import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
pTime = 0
cTime = 0
cap = cv2.VideoCapture(1)
detector = htm.handDetector()
while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=True )
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[4])
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    #cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
     #           (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
'''