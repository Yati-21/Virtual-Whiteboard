import numpy as np
import cv2
import time             #for fps
import os               #to access header files
import HandTracking_Yati as htm

#########
brushThickness = 15
eraserThickness = 70
########

#import images
folderPath = "Header"
myList = os.listdir(folderPath)   #dir - directory
#print(myList)   #list of images name   --we will use these images t overlay on top(next part)


overlayList = []
for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')   #read images - complete path
    overlayList.append(image)   #storing it in overlayList
#print(len(overlayList))

header = overlayList[0]   #default image on overlay
drawColor = (255,0,255)


#create loop and run webcam
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)   #set the size


detector = htm.handDetector(detectionCon=0.65)  #high confidence because we want it to be good in painting , we dont want mistakes here and there
xp, yp = (0,0)   #x previous and y previous --initialize (before x1, y1(current position))

#creating anew canvas on which we'll draw
imgCanvas = np.zeros((720,1280,3), np.uint8)


while True:
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img,1)          #flip orizontally becaude latterally inverted
    
    # 2. Find hand landmarks using Hand Tracking Module
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)
    if len(lmList) != 0:
        #print(lmList)
        x1, y1 = lmList[8][1:] #tip of index finger 
        x2, y2 = lmList[12][1:] #tip of middle finger 
        
    
        # 3. Check which fingers are up(because when index finger is up -- draw
        #                                       when 2 fingers up - no draw - only select)

        fingers =detector.fingersUp()
        print(fingers)
    
    
        # 4. if selection mode - 2 fingers are up then , select
        if fingers[1] and fingers[2]:
            xp, yp = 0,0
            #print("Selection Mode")
            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,cv2.FILLED)  #make a rectangle if selction mode between landmarks
        
            #check if we are at the top of the image , if yes then change the mode
            if y1<125:
                #clicking for the first colour
                if 250<x1<450 :
                    header = overlayList[0]
                    drawColor = (255,0,255)
                elif 550<x1<750 :   ##clicking for the second colour
                    header = overlayList[1]
                    drawColor = (255,0,0)
                elif 800<x1<950 :   ##clicking for the third colour
                    header = overlayList[2]
                    drawColor = (0,255,0)
                elif 1050<x1<1200 :   ##clicking for the erasee
                    header = overlayList[3]
                    drawColor = (0,0,0)   # black colour as eraser
            cv2.rectangle(img,(x1,y1-25),(x2,y2+25), drawColor,cv2.FILLED)

        # 5. if drawing mode - index finger is up 
        if fingers[1] and fingers[2]==False:
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)  #draw circle when drawing mode 
            #print("Drawing Mode")
            
            #drawing lines between the current point and the previous point
            if(xp == 0 and yp==0):   #i.e , Very first frame(just detected the hand) 
                                    #just started drawing , there must not be any line at that time
                                    # therefore we'll draw a point at that time
                xp,yp = x1,y1
            
            if drawColor == (0,0,0):
                cv2.line(img, (xp,yp),(x1,y1), drawColor, eraserThickness) 
                cv2.line(imgCanvas, (xp,yp),(x1,y1), drawColor, eraserThickness) 
            else:
                cv2.line(img, (xp,yp),(x1,y1), drawColor, brushThickness) 
                cv2.line(imgCanvas, (xp,yp),(x1,y1), drawColor, brushThickness) 
            
            xp,yp = x1,y1
            
            
        # # Clear Canvas when all fingers are up
        # if all (x >= 1 for x in fingers):
        #     imgCanvas = np.zeros((720, 1280, 3), np.uint8)
    
    
    imgGray = cv2.cvtColor(imgCanvas , cv2.COLOR_BGR2GRAY)
    _ , imgInv = cv2.threshold(imgGray , 50 , 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv ,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img , imgInv)
    img = cv2.bitwise_or(img , imgCanvas)
    
    
    # overlaying our image
    # imag - matrix , we just have to define where is the location
    # and we're going to slice itqq
    #setting the header image
    
    img[0:125, 0:1280] = header        # 0:125 --height    0:1280--width
    # img = cv2.addWeighted (img,0.5,imgCanvas, 0.5,0)     #drawing on original image
                                                           # agg/merge/blend the two images
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("Inv", imgInv)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

