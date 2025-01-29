import cv2
import numpy as np

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

# Variable to track if a circle is completed
circle_completed = False
circle_frames_count = 0

# Define the minimum radius to consider a circle as valid
MIN_CIRCLE_RADIUS = 60  # You can adjust this value

def is_circle_contour(contour):
    # Check if the contour forms a closed shape similar to a circle
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    return len(approx) > 8  # Circle-like if more than 8 vertices

def is_x_drawn(canvas):
    # Check if an 'X' shape is drawn on the canvas
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small objects
            rect = cv2.boundingRect(contour)
            aspect_ratio = rect[2] / rect[3]  # Width / Height
            if 0.8 <= aspect_ratio <= 1.2:  # Roughly square bounding box
                hull = cv2.convexHull(contour)
                if len(hull) >= 4:  # Check if there are at least 4 points in the contour
                    return True
    return False

def is_pen_horizontal(contour):
    # Check if the pen is in a horizontal position
    rect = cv2.minAreaRect(contour)
    (center, (width, height), angle) = rect
    aspect_ratio = max(width, height) / min(width, height)
    return aspect_ratio > 2.0  # Consider horizontal if the aspect ratio is greater than 2

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
        if cv2.contourArea(largest_contour) > 1000:  # Filter small objects
            # Check if the pen is in a horizontal position
            if is_pen_horizontal(largest_contour):
                # Get the center of the largest contour
                ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])

                    # Draw on the canvas if the pen is detected and horizontal
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (center_x, center_y), (0, 255, 0), 5)

                    prev_x, prev_y = center_x, center_y

        # Check if a circle-like contour is drawn on the canvas
        if is_circle_contour(largest_contour):
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            if radius > MIN_CIRCLE_RADIUS:  # Only save if the circle radius is large enough
                circle_completed = True

    else:
        prev_x, prev_y = None, None  # Reset if no pen is detected

    # Combine the canvas and the frame
    combined = cv2.add(frame, canvas)

    # Save the frame if a circle is completed
    if circle_completed:
        cv2.imwrite(f"circle_completed_frame_{circle_frames_count}.jpg", combined)
        circle_frames_count += 1  # Increment the counter for unique filenames
        circle_completed = False  # Reset the flag after saving

    # Clear the canvas if an 'X' is detected
     # if is_x_drawn(canvas):
        #canvas = np.zeros_like(frame)

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
