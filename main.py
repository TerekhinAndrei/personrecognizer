import cv2
import threading
from deepface import DeepFace
from queue import Queue
import numpy as np

# Global variable for storing the current frame
frame_queue = Queue(maxsize=10)  # Limit on the queue size to manage memory
result_queue = Queue()
current_results = []

def analyze_frame():
    temp_path = '/tmp/temp_frame.jpg'  # Change the path according to your OS
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Resize the frame for analysis
            frame_resized = cv2.resize(frame, (640, 480))
            cv2.imwrite(temp_path, frame_resized)
            
            try:
                # Perform analysis using DeepFace
                analysis = DeepFace.analyze(img_path=temp_path, actions=['gender', 'age'], enforce_detection=False)
                
                # Filter and format the analysis result
                gender_result = []
                age_result = None
                for result in analysis:
                    if 'gender' in result:
                        gender = result['gender']
                        # Get the gender with the highest confidence
                        max_gender = max(gender.items(), key=lambda x: x[1])
                        gender_result.append(f"Gender: {max_gender[0]} ({max_gender[1]:.2f})")
                    
                    if 'age' in result:
                        age = result['age']
                        age_result = f"Age: {age}"
                
                # Save only the most confident gender and age
                filtered_results = []
                if gender_result:
                    filtered_results.append(max(gender_result, key=lambda x: float(x.split('(')[-1].split(')')[0])))
                if age_result:
                    filtered_results.append(age_result)
                
                result_queue.put(filtered_results)
            except Exception as e:
                result_queue.put([f"Analysis error: {e}"])

def create_black_background(width, height):
    return np.zeros((height, width, 3), dtype=np.uint8)

def main():
    global current_results  # Declare that current_results is a global variable
    cap = cv2.VideoCapture(0)

    # Create a window and set it to resize mode
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    window_width = 640
    window_height = 480
    cv2.resizeWindow('Video', window_width, window_height)

    # Get camera dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return
    frame_height, frame_width = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (0, 255, 0)
    font_thickness = 2

    # Create and start a thread for analysis
    analysis_thread = threading.Thread(target=analyze_frame, daemon=True)
    analysis_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize the frame for display while maintaining aspect ratio
        if frame_width > frame_height:
            new_width = window_width
            new_height = int((window_width / frame_width) * frame_height)
        else:
            new_height = window_height
            new_width = int((window_height / frame_height) * frame_width)

        frame_resized = cv2.resize(frame, (new_width, new_height))
        # Create a black background for centering the image
        background = create_black_background(window_width, window_height)
        y_offset = (window_height - new_height) // 2
        x_offset = (window_width - new_width) // 2
        background[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = frame_resized

        # Try adding the frame to the analysis queue
        if frame_queue.qsize() < 10:  # Check queue size
            frame_queue.put(frame)

        # Get analysis results from the queue if available
        if not result_queue.empty():
            current_results = result_queue.get()

        # Display text from the analysis results
        y0, dy = 30, 30
        with result_queue.mutex:
            for i, text in enumerate(current_results):
                cv2.putText(background, text, (10, y0 + i * dy), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        # Display video
        cv2.imshow('Video', background)
        if cv2.waitKey(1) & 0xFF == 27:  # Esc key code - 27
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()