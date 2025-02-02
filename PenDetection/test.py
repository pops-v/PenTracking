from ultralytics import YOLO
import cv2

def test_model_video(video_path=0):  # Default is webcam (0)
    model = YOLO(r"C:/Users/pops/Desktop/PenDetection/runs/detect/train5/weights/best.pt")  # Load the trained model

    # Open the video capture (use 0 for webcam or specify video file path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from video.")
            break

        # Perform inference
        results = model(frame)

        # Since results is a list, access the first result
        result = results[0]  # Access the first result object from the list

        # Draw the results on the frame (plotting bounding boxes and labels)
        annotated_frame = result.plot()

        # Display the frame
        cv2.imshow("Detection Results", annotated_frame)

        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_model_video(0)  # Use 0 for webcam, or replace with video file path
