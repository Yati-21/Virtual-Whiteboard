while True:
    success, img = cap.read()  # Read a frame from the video capture
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR image to RGB
    results = hands.process(imageRGB)  # Process the frame to detect hands
    
    if results.multi_hand_landmarks:  # If hands are detected in the frame
        for handLms in results.multi_hand_landmarks:  # Iterate through each detected hand   
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  # Draw landmarks and connections
    
    cTime = time.time()  # Current time
    fps = 1 / (cTime - pTime)  # Calculate frames per second
    pTime = cTime  # Update previous time
    
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  # Display FPS
    cv2.imshow("Image", img)  # Display the annotated image
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop on 'q' key press
        break




import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)   #using webcam number 0

mpHands = mp.solutions.hands   #mp.solutions.hands refers to a module or sub-package within the MediaPipe library that 
                               #provides functionality for hand tracking and analysis.
                               #such as detecting hand landmarks (key points on the hand), estimating hand poses, and tracking hand movements.
hands = mpHands.Hands()     #craeting hands object!    #uses only RGB Image
mpDraw = mp.solutions.drawing_utils  #The drawing_utils module provides convenient functions to accomplish this. It typically includes functions to draw landmarks, connections between landmarks, bounding boxes, and other visual annotations.



#FOR  FPS (FPS or frame per second or frame rate can be defined as number of frames displayed per second.)

pTime = 0   #previous time
cTime = 0 #current time    


while True:
    success, img = cap.read()    #this will give us our frame
        # cap is an object created by OpenCV's VideoCapture class, representing a video file or a capturing device(webcam).
        # read() is a method of the VideoCapture object. 
        # When you call cap.read(), it reads the next frame from the video file or stream represented by the cap object.
        # The method returns two values:
        # The first value is a Boolean indicating whether a frame was successfully read. If a frame was read successfully, it returns True; otherwise, it returns False, indicating either the end of the video file or an error in reading the frame.
        # The second value is the actual frame read from the video source, typically represented as a NumPy array.
    
    #sending RGB image to the object    
    
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   #converting image to RGB because hands will only take RGB image
    results = hands.process(imageRGB)    #process the frame for us and give us the results
                                   #all we have to do is extract the info of results such as landmarks
    #print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks: #if true then for each handLandmark we'll draw the line using mpDra utilities function
                                    #results.multi_hand_landmarks is an attribute of the results object returned by the process() method of the MediaPipe Hands module. This attribute contains information about the detected hand landmarks for all hands found in the processed image or frame.
        for handLms in results.multi_hand_landmarks:     #handLms--- there could be multiple hands so, for each hand we'll draw the landmark points(red dots)
            
            #get info about each of the hand - id number and landmark info(x and y coordinate) for each hand i.e each handLms
            for id, lm in enumerate(handLms.landmark): #enumerate() is a built-in Python function that allows you to loop over an iterable (such as a list, tuple, or string) while also keeping track of the index of each item.
                
                #print(id, lm)   #lm are in decimal values
                
                # location must be in pixels 
                # we are getting the ratio of image
                # we are going to check the height, width and channel (below)
                
                h,w,c = img.shape
                cx, cy   = int(lm.x*w), int(lm.y*h) #center position (converting to integer by multiplying the values by height and width because lm.x and lm.y are in ratios)
                print(id, cx,cy)
                
                if id == 0:    #if landmark is 0(wrist) then we are making a circle to highlight it
                    cv2.circle(img, (cx,cy), 25, (255,0,255), cv2.FILLED)
                
            
            
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)    #we'll not draw on RGB image but on the original image
                                               # ^^^^^^^^^^^^^^^^^^^^^^ -- used to draw the lines(white lines) between the Landmark POints(red points)
     
     
    
    cTime = time.time()  #current time   time() function returns the number of seconds passed since epoch (the point where time begins)
    fps = 1/(cTime-pTime)    #For calculating FPS, we will be keeping the record of the time when last frame processed and the end time when current frame processed.
                             #So, the processing time for one frame will be time difference between current time and previous frame time .
    pTime= cTime
    
    
    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),3)   #putting Time on screen   
                                                                            #(int for rounding (we dont want decimal va;ues on screen))
                                                                                    # 3 - scale
                                                                                    #(255,0,255) - purple colour
                                                                                    # 3 - thickness
    cv2.imshow("Image", img)   
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
