import cv2
import numpy as np
import os
import json
import math

# File to save and load HSV values
HSV_VALUES_FILE = "hsv_values.json"

def nothing(x):
    pass

# Load HSV values from a file
def load_hsv_values():
    if os.path.exists(HSV_VALUES_FILE):
        with open(HSV_VALUES_FILE, 'r') as file:
            return json.load(file)
    return {
        "Lower-H": 61,
        "Lower-S": 47,
        "Lower-V": 104,
        "Upper-H": 140,
        "Upper-S": 255,
        "Upper-V": 255
    }

# Save HSV values to a file
def save_hsv_values(values):
    with open(HSV_VALUES_FILE, 'w') as file:
        json.dump(values, file)

# Load HSV values
hsv_values = load_hsv_values()

# Create a window for trackbars
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 600, 400)

# Add trackbars
cv2.createTrackbar("Lower-H", "Trackbars", hsv_values["Lower-H"], 179, nothing)
cv2.createTrackbar("Lower-S", "Trackbars", hsv_values["Lower-S"], 255, nothing)
cv2.createTrackbar("Lower-V", "Trackbars", hsv_values["Lower-V"], 255, nothing)
cv2.createTrackbar("Upper-H", "Trackbars", hsv_values["Upper-H"], 179, nothing)
cv2.createTrackbar("Upper-S", "Trackbars", hsv_values["Upper-S"], 255, nothing)
cv2.createTrackbar("Upper-V", "Trackbars", hsv_values["Upper-V"], 255, nothing)

# Feature toggles
cv2.createTrackbar("CLAHE", "Trackbars", 1, 1, nothing)
cv2.createTrackbar("Gaussian Blur", "Trackbars", 0, 1, nothing)
cv2.createTrackbar("Gaussian Kernel", "Trackbars", 5, 20, nothing)
cv2.createTrackbar("Top-hat", "Trackbars", 0, 1, nothing)
cv2.createTrackbar("Black-hat", "Trackbars", 0, 1, nothing)

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

canvas = None
prev_x, prev_y = None, None

# Function to apply Gaussian Blur
def apply_gaussian_blur(image, kernel_size=5):
    kernel_size = max(3, kernel_size // 2 * 2 + 1)  # Ensure kernel size is odd
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Function to apply Top-hat transformation
def apply_top_hat(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

# Function to apply Black-hat transformation
def apply_black_hat(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        if canvas is None:
            canvas = np.zeros_like(frame)

        # Get trackbar positions
        lower_h = cv2.getTrackbarPos("Lower-H", "Trackbars")
        lower_s = cv2.getTrackbarPos("Lower-S", "Trackbars")
        lower_v = cv2.getTrackbarPos("Lower-V", "Trackbars")
        upper_h = cv2.getTrackbarPos("Upper-H", "Trackbars")
        upper_s = cv2.getTrackbarPos("Upper-S", "Trackbars")
        upper_v = cv2.getTrackbarPos("Upper-V", "Trackbars")

        clahe_enabled = cv2.getTrackbarPos("CLAHE", "Trackbars")
        gaussian_blur_enabled = cv2.getTrackbarPos("Gaussian Blur", "Trackbars")
        gaussian_kernel_size = cv2.getTrackbarPos("Gaussian Kernel", "Trackbars")
        top_hat = cv2.getTrackbarPos("Top-hat", "Trackbars")
        black_hat = cv2.getTrackbarPos("Black-hat", "Trackbars")

        # Apply CLAHE
        if clahe_enabled:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Apply Gaussian Blur
        if gaussian_blur_enabled:
            frame = apply_gaussian_blur(frame, gaussian_kernel_size)

        # Apply Morphological Transformations
        if top_hat:
            frame = apply_top_hat(frame)
        if black_hat:
            frame = apply_black_hat(frame)

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([lower_h, lower_s, lower_v])
        upper_bound = np.array([upper_h, upper_s, upper_v])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:
                x, y, w, h = cv2.boundingRect(largest_contour)
                center_x, center_y = x + w // 2, y + h // 2
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

                # Draw on canvas
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (center_x, center_y), (255, 0, 0), 5)

                prev_x, prev_y = center_x, center_y
            else:
                prev_x, prev_y = None, None
        else:
            prev_x, prev_y = None, None

        # Combine frame with canvas
        combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

        cv2.imshow("Live Feed", combined)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):  # Clear the canvas when 'c' is pressed
            canvas = np.zeros_like(frame)

finally:
    # Save the HSV values before exiting
    hsv_values = {
        "Lower-H": cv2.getTrackbarPos("Lower-H", "Trackbars"),
        "Lower-S": cv2.getTrackbarPos("Lower-S", "Trackbars"),
        "Lower-V": cv2.getTrackbarPos("Lower-V", "Trackbars"),
        "Upper-H": cv2.getTrackbarPos("Upper-H", "Trackbars"),
        "Upper-S": cv2.getTrackbarPos("Upper-S", "Trackbars"),
        "Upper-V": cv2.getTrackbarPos("Upper-V", "Trackbars")
    }
    save_hsv_values(hsv_values)

    cap.release()
    cv2.destroyAllWindows()
