"""
Hand Tracking Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone
"""

import cv2
import mediapipe as mp
import time
import math
import numpy as np

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

def findHands(self, img, draw=True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if self.results.multi_hand_landmarks:
        for handLms in self.results.multi_hand_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)

    return img

def findPosition(self, img, handNo=0, draw=True):
    xList = []
    yList = []
    bbox = []
    self.lmList = []
    if self.results.multi_hand_landmarks:
        myHand = self.results.multi_hand_landmarks[handNo]
        for id, lm in enumerate(myHand.landmark):
            # print(id, lm)
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            xList.append(cx)
            yList.append(cy)
            # print(id, cx, cy)
            self.lmList.append([id, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

    xmin, xmax = min(xList), max(xList)
    ymin, ymax = min(yList), max(yList)
    bbox = xmin, ymin, xmax, ymax

    if draw:
        cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),(0, 255, 0), 2)

    return self.lmList, bbox

def fingersUp(self):
    fingers = []
    # Thumb
    if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Fingers
    for id in range(1, 5):
        if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

        # totalFingers = fingers.count(1)

    return fingers

def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
    x1, y1 = self.lmList[p1][1:]
    x2, y2 = self.lmList[p2][1:]
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
    cap = cv2.VideoCapture(1)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
        (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
    
    
    


################### EXPLAINATION  #############
'''
# Import required libraries
import cv2  # OpenCV library for computer vision tasks
import mediapipe as mp  # MediaPipe library for hand tracking and pose estimation
import time  # Library for time-related functions
import math  # Math library for mathematical operations
import numpy as np  # NumPy library for numerical computations

# Define a class for hand detection
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # Initialize the hand detector with specified parameters
        self.mode = mode  # Inference mode (whether to run in static image mode or video mode)
        self.maxHands = maxHands  # Maximum number of hands to detect
        self.detectionCon = detectionCon  # Minimum confidence threshold to consider detection successful
        self.trackCon = trackCon  # Minimum confidence threshold to consider tracking successful

        # Initialize MediaPipe Hands module
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.detectionCon, self.trackCon)

        # Initialize MediaPipe Drawing Utilities
        self.mpDraw = mp.solutions.drawing_utils

        # Define the IDs of finger tips
        self.tipIds = [4, 8, 12, 16, 20]

    # Method to find hands in an image
    def findHands(self, img, draw=True):
        """
        Find hands in the image and draw landmarks and connections on the image.

        Parameters:
            img (numpy.ndarray): Input image.
            draw (bool): Flag to specify whether to draw landmarks and connections on the image.
        
        Returns:
            img (numpy.ndarray): Image with landmarks and connections drawn (if draw=True).
        """
        # Convert the image to RGB format (required by MediaPipe)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the image to detect hands
        self.results = self.hands.process(imgRGB)

        # Draw landmarks and connections on the image if specified
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)

        return img

    # Method to find the positions of landmarks and bounding box of a hand
    def findPosition(self, img, handNo=0, draw=True):
        """
        Find the positions of landmarks and bounding box of a hand in the image.

        Parameters:
            img (numpy.ndarray): Input image.
            handNo (int): Index of the hand to find (default is 0 for the first detected hand).
            draw (bool): Flag to specify whether to draw landmarks and bounding box on the image.

        Returns:
            lmList (list): List of landmark positions.
            bbox (tuple): Bounding box coordinates (xmin, ymin, xmax, ymax).
        """
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        bbox = xmin, ymin, xmax, ymax

        if draw:
            cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),(0, 255, 0), 2)

        return self.lmList, bbox

    # Method to detect if fingers are up
    def fingersUp(self):
        """
        Detect if fingers are up based on the landmark positions.

        Returns:
            fingers (list): List indicating which fingers are up (1 for up, 0 for down).
        """
        fingers = []
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)  # Thumb
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)  # Other fingers
            else:
                fingers.append(0)

        return fingers

    # Method to find the distance between two points
    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        """
        Find the distance between two landmark points and optionally draw it on the image.

        Parameters:
            p1 (int): Index of the first landmark point.
            p2 (int): Index of the second landmark point.
            img (numpy.ndarray): Input image.
            draw (bool): Flag to specify whether to draw the distance line on the image.
            r (int): Radius of the circles representing the landmark points.
            t (int): Thickness of the line representing the distance.

        Returns:
            length (float): Distance between the two points.
            img (numpy.ndarray): Image with the distance line drawn (if draw=True).
            [x1, y1, x2, y2, cx, cy] (list): List containing coordinates of the two points and their midpoint.
        """
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

# Main function
def main():
    pTime = 0  # Previous time
    cTime = 0  # Current time

    # Capture video from the camera
    cap = cv2.VideoCapture(1)

    # Initialize the hand detector
    detector = handDetector()

    # Main loop to process frames
    while True:
        success, img = cap.read()  # Read a frame from the video
        img = detector.findHands(img)  # Find hands in the frame
        lmList, bbox = detector.findPosition(img)  # Find positions of landmarks and bounding box
        if len(lmList) != 0:
            print(lmList[4])  # Print the coordinates of a specific landmark (e.g., index finger tip)

        cTime = time.time()  # Get the current time
        fps = 1 / (cTime - pTime)  # Calculate frames per second
        pTime = cTime  # Update the previous time

        # Display FPS on the frame
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Display the frame
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()  # Call the main function
'''