import cv2
import numpy as np
import HandTrackingModule as htm
import ImageProcessingModule as ipm
import time
import autopy

FIST   = [0, 0, 0, 0, 0]
THUMB  = [1, 0, 0, 0, 0]
INDEX  = [0, 1, 0, 0, 0]
MIDDLE = [0, 0, 1, 0, 0]
RING   = [0, 0, 0, 1, 0]
PINKY  = [0, 0, 0, 0, 1]

INDEX_MIDDLE = [0, 1, 1, 0, 0]
INDEX_MIDDLE_RING = [0, 1, 1, 1, 0]

def main(mouseMovementSmoothening:int = 7, 
         fingersDistanceThreshold:list = [23, 37], 
         frameReduction:int = 90, 
         frameYDisplacement:int = 85,
         screenDisplay:bool = True):
    
    if mouseMovementSmoothening < 0:
        raise ValueError("mouseMovementSmoothening must be a positive value")
        
    if len(fingersDistanceThreshold) != 2:
        raise ValueError("fingersDistanceThreshold must be a list of only 2 integer values")
    
    if type(fingersDistanceThreshold[0]) != int or type(fingersDistanceThreshold[1]) != int:
        raise ValueError("fingersDistanceThreshold must have only integer values")
    
    if fingersDistanceThreshold[0] < 0 or fingersDistanceThreshold[1] < 0:
        raise ValueError("fingersDistanceThreshold must have only positive integer values")
    
    if fingersDistanceThreshold[0] > fingersDistanceThreshold[1]:
        raise ValueError("fingersDistanceThreshold min must be less than max")
    
    if frameYDisplacement > frameReduction:
        raise ValueError("frameYDisplacement must be less than frameReduction")
    
    pTime = 0
    cTime = 0
    clicked = False
    leftClickHold = False
    previousHandPosition = FIST

    wCam, hCam = 640, 480
    wScr, hScr = autopy.screen.size()
    
    
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0
    
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    
    processor = ipm.imageProcessor()
    detector = htm.handDetector(maxHands=1, detectionCon=0.6, trackCon=0.6)
    
    while True:
        success, img = cap.read()
        
        # 1: Blur Background for better detection
        img = processor.backgroungBlur(img, blur_type=0.3)
    
        # 2. Find Hand Landmarks
        img = detector.findHands(img, draw=screenDisplay)
        lmList, bbox = detector.findPosition(img, draw=screenDisplay)
        if screenDisplay:
            cv2.rectangle(img, 
                          (frameReduction, frameReduction - frameYDisplacement), 
                          (wCam - frameReduction, 
                           hCam - frameReduction - frameYDisplacement), 
                          (255, 0, 255), 2)
            
        # 3. Get the tip of index and middle fingers
        if len(lmList) != 0:
            # Thumb Finger
            xThumb, yThumb   = lmList[4][1:]
            # Index Finger
            xIndex, yIndex   = lmList[8][1:]
            # Middle Finger
            xMiddle, yMiddle = lmList[12][1:]
            # Ring Finger
            xRing, yRing     = lmList[16][1:]
            # Pinky Finger
            xPinky, yPinky   = lmList[20][1:]
            
            # a. Check which fingers are up
            fingers = detector.fingersUp()
            
            if fingers == FIST:
                previousHandPosition = FIST
                
            # b. Thumb : Left Click (Hold) Mode
            if fingers == THUMB:
                if screenDisplay:
                    cv2.circle(img, (xThumb, yThumb), 15, (0, 255, 0), cv2.FILLED)
            
                if previousHandPosition != THUMB:
                    if leftClickHold == False:
                        # i. First Left Click Hold
                        leftClickHold = True
                        autopy.mouse.toggle(autopy.mouse.Button.LEFT, down=leftClickHold)
                    else:
                        # ii. When Index Finger is up second time Release Hold
                        leftClickHold = False
                        autopy.mouse.toggle(autopy.mouse.Button.LEFT, down=leftClickHold)
                    
                
                previousHandPosition = THUMB
            
            # c. Only Index Finger : Moving Mode
            if fingers == INDEX:
                # i. Convert Coordinates
                x3 = np.interp(xIndex, (frameReduction, wCam - frameReduction), (0, wScr))
                y3 = np.interp(yIndex, 
                               (frameReduction - frameYDisplacement, 
                                hCam - frameReduction - frameYDisplacement), 
                               (0, hScr))
                    
                # ii. Smoothen Values
                clocX = plocX + (x3 - plocX) / mouseMovementSmoothening
                clocY = plocY + (y3 - plocY) / mouseMovementSmoothening
                
                # iii. Move Mouse
                moveX = wScr - clocX
                moveY = clocY
                
                if moveX < 0:
                    moveX = 0
                if moveX > wScr:
                    moveX = wScr
                
                if moveY < 0:
                    moveY = 0
                if moveY > hScr:
                    moveY = hScr
                    
                autopy.mouse.move(moveX, moveY)
                
                if screenDisplay:
                    cv2.circle(img, (xIndex, yIndex), 15, (255, 0, 255), cv2.FILLED)
                
                plocX, plocY = clocX, clocY

                previousHandPosition = INDEX
            
            # d. Both Index and Middle Fingers are up : Left Clicking Mode
            if fingers == INDEX_MIDDLE:
                # i. Find Distance between Fingers
                fingersDistance, img, lineInfo = detector.findDistance(8, 12, img, draw=screenDisplay)

                # ii. Click if distance is short
                if fingersDistanceThreshold[0] < fingersDistance < fingersDistanceThreshold[1]:
                    if screenDisplay:
                        cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                
                    if not clicked:
                        autopy.mouse.click(autopy.mouse.Button.LEFT)
                        clicked = True
                        
                elif fingersDistance > fingersDistanceThreshold[1]:
                    clicked = False
                
                previousHandPosition = INDEX_MIDDLE
                    
            # e. All Index, Middle, and Ring Fingers are up : Right Clicking Mode
            if fingers == INDEX_MIDDLE_RING:
                # i. Find Distance between Fingers
                fingersDistance, img, lineInfo = detector.findDistance(8, 12, img, draw=screenDisplay)

                # ii. Click if distance is short
                if fingersDistanceThreshold[0] < fingersDistance < fingersDistanceThreshold[1]:
                    if screenDisplay:
                        cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                
                    if not clicked:
                        autopy.mouse.click(autopy.mouse.Button.RIGHT)
                        clicked = True
                elif fingersDistance > fingersDistanceThreshold[1]:
                    clicked = False
                
                previousHandPosition = INDEX_MIDDLE_RING
                        
        if screenDisplay:
            # 4. Frame Rate
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
            # 5. Display
            cv2.imshow('Image', img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        
if __name__ == '__main__':
    main(screenDisplay=False)
