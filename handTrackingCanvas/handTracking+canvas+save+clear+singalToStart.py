import cv2
import numpy as np
import mediapipe as mp
import os
import time

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize the drawing canvas
canvas = None

# Distance threshold for detecting fingers close together
touch_threshold = 30

# Start capturing video from the webcam
cap = cv2.VideoCapture(1)

# Initialize the toggle state and detection variables
fist_detected_once = False
fist_detected = False

# Initialize variables to store previous positions for drawing
prev_x, prev_y = None, None

# Timer to detect if hand is no longer in the frame
last_hand_visible_time = time.time()
hand_in_frame = False
hand_invisibility_threshold = 2  # seconds before saving after hand leaves

# Flag to check if the frame has been saved after hand is out
frame_saved = False

# Directory to save frames
save_dir = r"C:\Users\pops\Desktop\openCV_Advanced\handTrackingCanvas\images"

# Ensure the directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    if canvas is None:
        canvas = np.zeros_like(frame)  # Initialize a blank canvas with the same size as the frame

    # Convert the frame to RGB (required by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Check if a hand is detected
    if results.multi_hand_landmarks:
        hand_in_frame = True
        last_hand_visible_time = time.time()  # Reset the timer when hand is detected
        frame_saved = False  # Reset the frame saved flag when the hand is detected again

        for hand_landmarks in results.multi_hand_landmarks:
            # Get the positions of the finger tips: Thumb(4), Index(8), Middle(12)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Get the x, y coordinates of the finger tips
            thumb_tip_coords = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
            index_tip_coords = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
            middle_tip_coords = (int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0]))
            ring_tip_coords = (int(ring_tip.x * frame.shape[1]), int(ring_tip.y * frame.shape[0]))
            pinky_tip_coords = (int(pinky_tip.x * frame.shape[1]), int(pinky_tip.y * frame.shape[0]))

            # Calculate the distances between the finger tips
            dist_thumb_index = np.linalg.norm(np.array(thumb_tip_coords) - np.array(index_tip_coords))
            dist_index_middle = np.linalg.norm(np.array(index_tip_coords) - np.array(middle_tip_coords))
            dist_ring_pinky = np.linalg.norm(np.array(ring_tip_coords) - np.array(pinky_tip_coords))
            dist_thumb_pinky = np.linalg.norm(np.array(thumb_tip_coords) - np.array(pinky_tip_coords))

            # Check if all five fingertips are close together (indicating a fist)
            if (dist_thumb_index < touch_threshold and
                dist_index_middle < touch_threshold and
                dist_ring_pinky < touch_threshold and
                dist_thumb_pinky < touch_threshold):
                
                # Toggle the state when a fist is detected
                if not fist_detected:
                    fist_detected_once = not fist_detected_once  # Toggle the state
                    fist_detected = True  # Mark fist as detected to avoid repeated toggling
            else:
                fist_detected = False  # Reset fist detection state when fingers are no longer close

            # Display fist detection status on the screen
            status_text = "Drawing: ON" if fist_detected_once else "Drawing: OFF"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (0, 255, 0) if fist_detected_once else (0, 0, 255), 2)

            # If drawing mode is ON, draw on the canvas
            if fist_detected_once:
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), thumb_tip_coords, (0, 255, 0), 2)
                prev_x, prev_y = thumb_tip_coords
            else:
                prev_x, prev_y = None, None  # Reset previous positions when not drawing

            # Optional: Visualize the finger tips on the frame
            cv2.circle(frame, thumb_tip_coords, 5, (255, 0, 0), -1)  # Thumb tip
            cv2.circle(frame, index_tip_coords, 5, (0, 255, 0), -1)  # Index tip
            cv2.circle(frame, middle_tip_coords, 5, (0, 0, 255), -1)  # Middle tip

    else:
        hand_in_frame = False

    # Combine the canvas and the frame
    combined = cv2.add(frame, canvas)

    # Check if the hand has been out of the frame for the defined threshold
    if not hand_in_frame and (time.time() - last_hand_visible_time) > hand_invisibility_threshold and not frame_saved:
        # Save the frame when the hand is no longer in the frame for the threshold time
        frame_filename = os.path.join(save_dir, f"no_hand_in_frame_{int(time.time())}.jpg")
        cv2.imwrite(frame_filename, combined)

        # Mark the frame as saved
        frame_saved = True

        # Reset the drawing canvas
        canvas = np.zeros_like(frame)
        prev_x, prev_y = None, None

    # Show the results
    cv2.imshow("Live Feed", combined)

    # Exit on pressing 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
