'''
import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2,detectionCon=0.5, trackCon=0.5):              #similar to constructor
        
        #self--obejct
        
        ## Initailiation :--
        self.mode = mode
        self.maxHands = maxHands
        # self.detectionCon = detectionCon
        # self.trackCon = trackCon
        self.config = {
            'min_detection_confidence': detectionCon,
            'min_tracking_confidence': trackCon
        }

        self.mpHands = mp.solutions.hands
        # self.hands = self.mpHands.Hands(self.mode, self.maxHands,float(self.detectionCon), float(self.trackCon))
        # self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, **self.config)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4,8,12,16,20]

    #hand detect
    def findHands(self, img, draw=True):    #img as a parameter - to find hands on
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:              #if draw is true:::i.e if we want to draw the had sthen only t will draw
                    self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)   #lines between the landmarks
        return img


    #to get the landmark values
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []  #lANDMARK LIST WHICH WE WILL RETURN
        if self.results.multi_hand_landmarks:   # to detect whether any hand was detected or not
            myHand = self.results.multi_hand_landmarks[handNo]   #which hand are we talking about
            
            
            #return landmark for the specified hand ::
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw: 
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()   #creating object - no need to pass arguments because default paraeters are already defined in __init__
    while True:
        success, img = cap.read()
        img = detector.findHands(img)   #calling method of class
        lmList = detector.findPosition(img)
        if len(lmList) != 0:   #if landmark list returned by findPosition is not empty then it will givie us the position of landmark 4(thumb)
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()

'''


############# EXPLAINATION  #################


import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # Initialization method, called when an object is created
        self.mode = mode  # Whether to detect hands in the static image(false) or track hands across different frames(true)
        self.maxHands = maxHands  # Maximum number of hands to detect
        # Configuration dictionary for minimum confidence levels for detection and tracking
        self.config = {
            'min_detection_confidence': detectionCon,   # Minimum confidence level for hand detection
            'min_tracking_confidence': trackCon        # Minimum confidence level for hand tracking
        }
        
        # Importing MediaPipe's hand model and drawing utilities
        self.mpHands = mp.solutions.hands
        # Creating an instance of the hand detection model
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, **self.config)
        # Importing drawing utilities for landmarks and connections
        self.mpDraw = mp.solutions.drawing_utils
        
        self.tipIds = [4,8,12,16,20]

    def findHands(self, img, draw=True):
        """This method takes an image (img) as input and detects hands in the image. 
        It converts the image to RGB format, processes it using the hand detection model, and 
        draws landmarks and connections on the image if draw is set to True. """
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB format
        self.results = self.hands.process(imgRGB)  # Process the image with the hand detection model
        
        if self.results.multi_hand_landmarks:  # If hands are detected in the image
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Draw landmarks and connections on the image
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img  # Return the image with landmarks drawn

    def findPosition(self, img, handNo=0, draw=True):
        """  This method finds the positions of landmarks (key points) on the detected hands. 
        It returns a list of landmark positions for a specified hand (handNo) in the image. """
        
        self.lmList = []  # Initialize a list to store landmark positions
        if self.results.multi_hand_landmarks:  # If hands are detected in the image
            myHand = self.results.multi_hand_landmarks[handNo]  # Get landmarks of the specified hand
            
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # Calculate pixel positions of landmarks
                self.lmList.append([id, cx, cy])  # Append landmark id and positions to the list
                if draw:
                    # Draw a circle at the landmark position on the image
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return self.lmList  # Return the list of landmark positions
    
    
    def fingersUp(self):
        fingers= []
        
        #Thumb
        if self.lmList[self.tipIds[0]][1]  < self.lmList[self.tipIds[0]-1][1] :   #check if tip of the thumb is on right or left
            fingers.append(1)
        else:
            fingers.append(0)
            
        #4 fingers
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2]  < self.lmList[self.tipIds[id]-2][2]:    #check if tip of the finger is on up or down
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
    
    
def main():
    """  This function is the main entry point of the script. 
    It initializes variables for tracking FPS, captures video from the default 
    camera (cap), and creates an instance of the handDetector class (detector). 
    Inside the main loop: 
    - It reads frames from the camera and passes them to the findHands method to detect hands and draw landmarks on the frames.
    - It calls the findPosition method to get the positions of landmarks, prints the position of the thumb if available.
    - It calculates and displays the frames per second (FPS) on the video feed.
    - It displays the processed image with landmarks and FPS on the screen using OpenCV's imshow function.
    - It exits the loop if the 'q' key is pressed.
    """
    pTime = 0  # Previous time for calculating FPS
    cTime = 0  # Current time
    
    cap = cv2.VideoCapture(0)  # Capture video from the default camera
    
    detector = handDetector()  # Create an instance of the handDetector class
    
    while True:
        success, img = cap.read()  # Read a frame from the camera
        img = detector.findHands(img)  # Find hands in the image
        lmList = detector.findPosition(img)  # Find landmark positions
        
        if len(lmList) != 0:  # If landmark list is not empty
            print(lmList[4])  # Print the position of the thumb
        
        cTime = time.time()  # Get current time
        fps = 1 / (cTime - pTime)  # Calculate frames per second
        pTime = cTime  # Update previous time
        
        # Display FPS on the image
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        cv2.imshow("Image", img)  # Display the image
        if cv2.waitKey(1) & 0xFF == ord('q'):  # If 'q' key is pressed, exit the loop
            break

if __name__ == "__main__":
    """ This block ensures that the main function is executed only when 
    the script is run directly, not when it's imported as a module.  """
    
    main()  # Call the main function
