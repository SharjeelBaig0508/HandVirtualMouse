import cv2
import time
import math
import numpy as np

class imageProcessor():
    def __init__(self):
        pass
        
    def backgroundBlur(self, img, blur_type=1):
        # Step 1: Convert to the HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Step 2: Create a mask based on medium to high Saturation and Value
        # - These values can be changed (the lower ones) to fit your environment
        mask = cv2.inRange(hsv, (0, 75, 40), (180, 255, 255))
        
        # Step 3: We need a to copy the mask 3 times to fit the frames
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        # Step 4: Create a blurred image using Gaussian blur
        blurred_image = cv2.GaussianBlur(img, (25, 25), blur_type)
        
        # Step 5: Combine the original with the blurred image based on mask
        img = np.where(mask_3d == (255, 255, 255), img, blurred_image)
        
        # Return Converted Img
        return img

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    processor = imageProcessor()
    
    while True:
        success, img = cap.read()
        img = processor.backgroundBlur(img)
            
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
if __name__ == '__main__':
    main()
