import cv2
import mediapipe as mp
import numpy as np
import time
import HandTrackingModule as htm
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

pTime = 0

# Get images from a folder to overlay
folderPath = "FingerImages"
myList = os.listdir(folderPath)
myList.sort()
overlaysList = []
for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    overlaysList.append(image)

detector = htm.HandDetector(detection_con=0.7,max_hands=1)

tipIds = [4, 8, 12, 16, 20]


while True:
    ret, frame = cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    fingers = []
    if len(lmList) != 0:
        # Thumb
        if lmList[1][1] < lmList[17][1]:
            # Left Hand
            if lmList[4][1] < lmList[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if lmList[4][1] > lmList[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Fingers
        for ids in range(1, len(tipIds)):
            if lmList[tipIds[ids]][2] < lmList[tipIds[ids]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)
        h, w, c = overlaysList[totalFingers].shape
        frame[0:h, 0:w] = overlaysList[totalFingers]
        cv2.rectangle(frame, (20, 300), (130, 400), (0, 255, 0), -1)
        cv2.putText(frame, str(totalFingers), (40, 380), cv2.FONT_HERSHEY_PLAIN,5, (255,255,255),
                    5)
    # print(lmList)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, "FPS: " + str(int(fps)), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 255), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
