import cv2
import mediapipe as mp
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations (optional)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Suppress Mediapipe warnings

# For FPS Calculation
prev_time = 0
cap = cv2.VideoCapture(0)

# Initialize Camera and Mediapipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


while True: 
    success, img = cap.read()
    imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results= hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
             for handLms in results.multi_hand_landmarks:
                for id,lm in enumerate(handLms.landmark):
                    #print(id, lm)
                    h, w, c =img.shape
                    cx,cy= int(lm.x*w), int(lm.y*h)
                    print(id,cx,cy)
                    if id==4:
                        cv2.circle(img, (cx, cy), 15, (255,0 ,255),cv2.FILLED)
            
             mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
     
      # Calculate and Display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)        
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



