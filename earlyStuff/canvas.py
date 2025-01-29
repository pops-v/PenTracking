import cv2
import numpy as np
from tkinter import Tk, Canvas, Button, colorchooser
import threading

# Live feed with OpenCV
def live_feed():
    def measure_objects(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small objects
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

                # Calculate dimensions
                width = int(rect[1][0])
                height = int(rect[1][1])
                cv2.putText(frame, f"{width}x{height} px", (int(rect[0][0]), int(rect[0][1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = measure_objects(frame)
        cv2.imshow('Live Feed - Press Q to Quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Drawing Canvas with Tkinter
def drawing_canvas():
    def choose_color():
        color_code = colorchooser.askcolor(title="Choose a color")[1]
        canvas.color = color_code

    def paint(event):
        x1, y1 = event.x - 2, event.y - 2
        x2, y2 = event.x + 2, event.y + 2
        canvas.create_oval(x1, y1, x2, y2, fill=canvas.color, outline=canvas.color)

    root = Tk()
    root.title("Drawing Canvas")

    canvas = Canvas(root, bg="white", width=800, height=600)
    canvas.pack()
    canvas.color = "black"

    canvas.bind("<B1-Motion>", paint)

    Button(root, text="Choose Color", command=choose_color).pack(side="left")
    Button(root, text="Clear", command=lambda: canvas.delete("all")).pack(side="left")

    root.mainloop()

# Run both functionalities
if __name__ == "__main__":
    # Use threads to run both functionalities independently
    threading.Thread(target=live_feed).start()
    drawing_canvas()
