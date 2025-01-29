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
# Distance between tips (2D)
touch_threshold = 30
# Depth threshold (z-axis)
depth_threshold = 0.05

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

fist_detected_once = False

# Initialize variables to store previous positions for drawing
prev_x, prev_y = None, None

# State variable to check if drawing is occurring
drawing_started = False

# Timer to detect if hand is no longer in the frame (e.g., 2 seconds)
last_hand_visible_time = time.time()
hand_in_frame = False
hand_invisibility_threshold = 2  # seconds before saving after hand leaves

# Flag to check if the frame has been saved after hand is out
frame_saved = False

# Directory to save frames
save_dir = r"C:\Users\pops\Desktop\openCV_Advanced\canvas+pen+save"

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

            # Calculate the distance between the tips of the thumb and index fingers
            dist_thumb_index = np.linalg.norm(np.array(thumb_tip_coords) - np.array(index_tip_coords))

            # Get the z-coordinates of the finger tips
            z_diff_thumb_index = abs(thumb_tip.z - index_tip.z)
            z_diff_index_middle = abs(index_tip.z - middle_tip.z)

            # Depth-based logic
            fingers_close_in_depth = z_diff_thumb_index < depth_threshold and z_diff_index_middle < depth_threshold

            # Check if all five fingertips are close together (indicating a fist)
            if (dist_thumb_index < touch_threshold and fingers_close_in_depth):
                if fist_detected_once:
                    fist_detected_once = False
                else:
                    fist_detected_once = True

                fist_detected = True
            else:
                fist_detected = False

            # Check if the thumb, index, and middle finger tips are close to each other (within the threshold)
            if fist_detected_once and dist_thumb_index < touch_threshold and fingers_close_in_depth:
                drawing_started = True
            else:
                drawing_started = False

            # Display fist detection status on the screen
            status_text = "Fist Detected" if fist_detected else "No Fist Detected"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if fist_detected else (0, 255, 0), 2)

            # If drawing has started, draw a line from the previous tip position to the current one
            if drawing_started:
                # Draw on the canvas if the pen is detected and drawing is active
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), thumb_tip_coords, (0, 255, 0), 2)

                # Update the previous positions
                prev_x, prev_y = thumb_tip_coords

            # Optional: Show the finger tips on the frame for visualization
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

        # Optionally, reset the drawing canvas if needed, or leave it as is
        canvas = np.zeros_like(frame)  # Uncomment this if you want to clear the canvas after saving
        prev_x, prev_y = None, None  # Reset the previous positions if needed

    # Show the results
    cv2.imshow("Live Feed", combined)

    # Exit on pressing 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
