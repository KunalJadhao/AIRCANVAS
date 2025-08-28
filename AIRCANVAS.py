import cv2
import mediapipe as mp
import numpy as np

class AirCanvas:
    def __init__(self):
        # Initialize MediaPipe Hands with adjusted parameters
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,  # Try different complexity levels
            min_detection_confidence=0.5,  # Lowered for better detection
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize drawing parameters
        self.canvas = None
        self.prev_x, self.prev_y = 0, 0
        self.color_index = 0
        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # Red, Green, Blue, Yellow
        self.brush_thickness = 5
        self.eraser_thickness = 20
        
        # Initialize webcam with error handling
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot access camera")
            exit()
            
        self.cap.set(3, 1280)  # Width
        self.cap.set(4, 720)   # Height
        
        # Create a window
        cv2.namedWindow('Air Canvas')
        
    def draw_color_options(self, frame):
        # Draw color selection boxes
        for i, color in enumerate(self.colors):
            cv2.rectangle(frame, (10, 10 + i*40), (50, 50 + i*40), color, -1)
        
        # Draw eraser option
        cv2.rectangle(frame, (10, 10 + len(self.colors)*40), 
                     (50, 50 + len(self.colors)*40), (255, 255, 255), -1)
        cv2.putText(frame, "Eraser", (60, 40 + len(self.colors)*40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Highlight selected color
        cv2.rectangle(frame, (5, 5 + self.color_index*40), 
                     (55, 55 + self.color_index*40), (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
                
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Initialize canvas if not done yet
            if self.canvas is None:
                self.canvas = np.zeros_like(frame)
            
            # Convert frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # To improve performance, optionally mark the image as not writeable to pass by reference
            rgb_frame.flags.writeable = False
            results = self.hands.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            # Draw color options
            frame = self.draw_color_options(frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
                    
                    # Get index finger tip coordinates
                    h, w, c = frame.shape
                    x = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
                    y = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
                    
                    # Check if finger is in color selection area
                    if x < 60:
                        for i in range(len(self.colors) + 1):
                            if 10 + i*40 <= y <= 50 + i*40:
                                self.color_index = i
                                break
                    
                    # Draw on canvas
                    if self.prev_x == 0 and self.prev_y == 0:
                        self.prev_x, self.prev_y = x, y
                    
                    if self.color_index < len(self.colors):  # Regular drawing
                        cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), 
                                self.colors[self.color_index], self.brush_thickness)
                    else:  # Eraser mode
                        cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), 
                                (0, 0, 0), self.eraser_thickness)
                    
                    self.prev_x, self.prev_y = x, y
            else:
                self.prev_x, self.prev_y = 0, 0
            
            # Combine frame and canvas
            frame = cv2.add(frame, self.canvas)
            
            # Add instructions
            h, w, _ = frame.shape
            cv2.putText(frame, "Air Canvas - Move your index finger to draw", 
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'c' to clear, 'q' to quit", 
                       (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Air Canvas', frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.canvas = np.zeros_like(frame)  # Clear canvas
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    air_canvas = AirCanvas()
    air_canvas.run()