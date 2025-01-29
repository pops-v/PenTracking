import cv2
import mediapipe as mp

# Suppress TensorFlow INFO and WARNING logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress Mediapipe warnings
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Initialize MediaPipe FaceMesh
mpFaceMesh = mp.solutions.face_mesh
face_mesh = mpFaceMesh.FaceMesh()
mpDraw = mp.solutions.drawing_utils

# Initialize the camera
cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from camera")
        break

    # Convert BGR image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image to find face landmarks
    results = face_mesh.process(imgRGB)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the face mesh
            mpDraw.draw_landmarks(img, face_landmarks, mpFaceMesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                                  connection_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1))

    # Display the image with face mesh
    cv2.imshow('Face Mesh', img)

    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
