import cv2
import numpy as np
import os
import time  # Import time module for delays

def nothing(x):
    pass

# Create a window for trackbars to tune HSV values
cv2.namedWindow("Trackbars")
cv2.createTrackbar("Lower-H", "Trackbars", 100, 179, nothing)
cv2.createTrackbar("Lower-S", "Trackbars", 150, 255, nothing)
cv2.createTrackbar("Lower-V", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("Upper-H", "Trackbars", 140, 179, nothing)
cv2.createTrackbar("Upper-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Upper-V", "Trackbars", 255, 255, nothing)

# Initialize the canvas for drawing
canvas = None

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Initialize variables to store the pen's previous position
prev_x, prev_y = None, None

# Variable to track if a circle-ish shape is detected
circle_detected = False
circle_frames_count = 0

# Directory to save frames
save_dir = r"C:\Users\pops\Desktop\openCV_Advanced\canvas+pen+save"

# Ensure the directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Function to detect a circle-ish (oval) shape using contour fitting
def detect_oval_or_circle(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # Threshold to get binary image
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small contours
            # Fit an ellipse to the contour
            if len(contour) >= 5:  # At least 5 points are needed to fit an ellipse
                ellipse = cv2.fitEllipse(contour)
                (center, axes, angle) = ellipse
                aspect_ratio = float(axes[0]) / axes[1]  # Aspect ratio of the ellipse (width / height)

                # Check if the aspect ratio is close to 1 (circle-ish)
                if 0.5 <= aspect_ratio <= 1.2:  # Flexible for ovals
                    return True, ellipse  # Return if a valid circle-ish shape is found
    return False, None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    if canvas is None:
        canvas = np.zeros_like(frame)  # Initialize a blank canvas with the same size as the frame

    # Get the HSV range values from the trackbars
    lower_h = cv2.getTrackbarPos("Lower-H", "Trackbars")
    lower_s = cv2.getTrackbarPos("Lower-S", "Trackbars")
    lower_v = cv2.getTrackbarPos("Lower-V", "Trackbars")
    upper_h = cv2.getTrackbarPos("Upper-H", "Trackbars")
    upper_s = cv2.getTrackbarPos("Upper-S", "Trackbars")
    upper_v = cv2.getTrackbarPos("Upper-V", "Trackbars")

    # Define the lower and upper bounds for the pen's color
    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask to detect the pen's color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Reduce noise in the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:  # Filter small objects
            # Get the center and radius of the largest contour
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])

                # Draw on the canvas if the pen is detected
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (center_x, center_y), (0, 255, 0), 5)

                prev_x, prev_y = center_x, center_y

        # Detect circle-ish or oval shape on the canvas
        is_circle, ellipse = detect_oval_or_circle(canvas)
        if is_circle:
            # Draw the detected ellipse on the frame
            cv2.ellipse(frame, ellipse, (0, 255, 0), 2)  # Draw the ellipse in green
            circle_detected = True

    else:
        prev_x, prev_y = None, None  # Reset if no pen is detected

    # Combine the canvas and the frame
    combined = cv2.add(frame, canvas)

    # Save the frame if a circle-ish shape is detected
    if circle_detected:
        # Introduce a delay of 1 second before saving and clearing the canvas
        time.sleep(1)  # Delay for 1 second

        # Save the image with a unique filename in the specified directory
        frame_filename = os.path.join(save_dir, f"circle_detected_frame_{circle_frames_count}.jpg")
        cv2.imwrite(frame_filename, combined)
        circle_frames_count += 1  # Increment the counter for unique filenames
        circle_detected = False  # Reset the flag after saving

        # Clear the canvas immediately after saving the frame
        canvas = np.zeros_like(frame)

    # Show the results
    cv2.imshow("Live Feed", combined)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):  # Clear the canvas when 'c' is pressed
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()
