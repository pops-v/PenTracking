import cv2
import numpy as np
import mediapipe as mp
import os
import time
from datetime import datetime

class HandDrawing:
    def __init__(self):
        # Initialize MediaPipe Hands model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Limit to one hand for better performance
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Drawing settings
        self.colors = {
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'blue': (255, 0, 0)
        }
        self.current_color = self.colors['green']
        self.brush_thickness = 2
        
        # Initialize variables
        self.canvas = None
        self.prev_x = self.prev_y = None
        self.touch_threshold = 120
        self.recent_points = []
        self.smooth_factor = 5
        
        # Timer settings
        self.last_hand_visible_time = time.time()
        self.hand_in_frame = False
        self.frame_saved = False
        self.hand_invisibility_threshold = 2
        
        # Save directory setup
        self.save_dir = os.path.join(os.path.expanduser("~"), "Desktop", "HandDrawings")
        os.makedirs(self.save_dir, exist_ok=True)

    def process_gestures(self, hand_landmarks, frame_shape):
        """Process hand gestures to determine drawing mode and color"""
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        # Convert coordinates
        index_tip_coords = (int(index_tip.x * frame_shape[1]), int(index_tip.y * frame_shape[0]))
        thumb_tip_coords = (int(thumb_tip.x * frame_shape[1]), int(thumb_tip.y * frame_shape[0]))
        middle_tip_coords = (int(middle_tip.x * frame_shape[1]), int(middle_tip.y * frame_shape[0]))
        
        # Calculate distances
        thumb_index_dist = np.linalg.norm(np.array(index_tip_coords) - np.array(thumb_tip_coords))
        middle_index_dist = np.linalg.norm(np.array(index_tip_coords) - np.array(middle_tip_coords))
        
        # Color change gesture (thumb and index pinch)
        if thumb_index_dist < 30:
            self.cycle_color()
            return False, index_tip_coords
            
        # Clear canvas gesture (middle and index pinch)
        if middle_index_dist < 30:
            self.clear_canvas()
            return False, index_tip_coords
            
        # Normal drawing mode
        return True, index_tip_coords

    def cycle_color(self):
        """Cycle through available colors"""
        colors_list = list(self.colors.values())
        current_index = colors_list.index(self.current_color)
        self.current_color = colors_list[(current_index + 1) % len(colors_list)]

    def clear_canvas(self):
        """Clear the drawing canvas"""
        if self.canvas is not None:
            self.canvas.fill(0)

    def save_drawing(self, frame):
        """Save the current drawing with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.save_dir, f"drawing_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        return filename

    def run(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            if self.canvas is None:
                self.canvas = np.zeros_like(frame)

            # Process frame with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                self.hand_in_frame = True
                self.last_hand_visible_time = time.time()
                self.frame_saved = False

                # Process the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                drawing_mode, index_tip_coords = self.process_gestures(hand_landmarks, frame.shape)

                if drawing_mode:
                    # Smooth drawing
                    self.recent_points.append(index_tip_coords)
                    if len(self.recent_points) > self.smooth_factor:
                        self.recent_points.pop(0)

                    smooth_x = int(np.mean([p[0] for p in self.recent_points]))
                    smooth_y = int(np.mean([p[1] for p in self.recent_points]))

                    if self.prev_x is not None and self.prev_y is not None:
                        cv2.line(self.canvas, (self.prev_x, self.prev_y), 
                                (smooth_x, smooth_y), self.current_color, self.brush_thickness)

                    self.prev_x, self.prev_y = smooth_x, smooth_y
                else:
                    self.prev_x = self.prev_y = None

                # Visualize index fingertip
                cv2.circle(frame, index_tip_coords, 5, self.current_color, -1)

            else:
                self.hand_in_frame = False
                if (time.time() - self.last_hand_visible_time) > self.hand_invisibility_threshold and not self.frame_saved:
                    saved_file = self.save_drawing(cv2.add(frame, self.canvas))
                    print(f"Drawing saved to: {saved_file}")
                    self.frame_saved = True
                    self.canvas.fill(0)
                    self.prev_x = self.prev_y = None

            # Display current color indicator
            cv2.circle(frame, (30, 30), 15, self.current_color, -1)
            
            # Combine canvas and frame
            combined = cv2.add(frame, self.canvas)
            cv2.imshow("Hand Drawing", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = HandDrawing()
    app.run()