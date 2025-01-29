import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize the drawing canvas
canvas = None

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Initialize variables to store previous positions for drawing
prev_x, prev_y = None, None

# State variable to check if drawing should start
drawing_started = False

# Threshold for determining if fingers are touching (distance)
touch_threshold = 50  # Adjust this based on your needs

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

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the positions of the finger tips: Thumb(4), Index(8), Middle(12)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Get the x, y coordinates of the finger tips
            thumb_tip_coords = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
            index_tip_coords = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
            middle_tip_coords = (int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0]))

            # Calculate the distance between the tips of the thumb, index, and middle finger
            dist_thumb_index = np.linalg.norm(np.array(thumb_tip_coords) - np.array(index_tip_coords))
            dist_index_middle = np.linalg.norm(np.array(index_tip_coords) - np.array(middle_tip_coords))

            # Check if the thumb, index, and middle finger tips are close to each other (within the threshold)
            if dist_thumb_index < touch_threshold and dist_index_middle < touch_threshold:
                drawing_started = True
            else:
                drawing_started = False

            # If drawing has started, draw a line from the previous tip position to the current one
            if drawing_started:
                # Draw on the canvas if the pen is detected and the drawing has started
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), thumb_tip_coords, (0, 255, 0), 5)

                # Update the previous positions
                prev_x, prev_y = thumb_tip_coords

            # Optional: Show the finger tips on the frame for visualization
            cv2.circle(frame, thumb_tip_coords, 10, (255, 0, 0), -1)  # Thumb tip
            cv2.circle(frame, index_tip_coords, 10, (0, 255, 0), -1)  # Index tip
            cv2.circle(frame, middle_tip_coords, 10, (0, 0, 255), -1)  # Middle tip

    # Combine the canvas and the frame
    combined = cv2.add(frame, canvas)

    # Show the results
    cv2.imshow("Live Feed", combined)

    # Exit on pressing 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
