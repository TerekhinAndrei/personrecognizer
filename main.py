import cv2
import threading
from deepface import DeepFace
from queue import Queue
import numpy as np
from playsound import playsound
import logging
import collections
from collections import Counter

logging.basicConfig(level=logging.DEBUG)

isStartVisualise = False
def set_visualise_true():
    global isStartVisualise
    isStartVisualise = True
    # Play sound only once
    if not sound_played_event.is_set():
        threading.Thread(target=play_sound, daemon=True).start()
        sound_played_event.set()

# Queues for storing results
gender_queue = collections.deque(maxlen=10)
age_queue = collections.deque(maxlen=10)
emotion_queue = collections.deque(maxlen=10)

# List of models
models = [
    "VGG-Face", 
    "Facenet", 
    "Facenet512", 
    "OpenFace", 
    "DeepFace", 
    "DeepID", 
    "ArcFace", 
    "Dlib", 
    "SFace",
    "GhostFaceNet",
]

# Global variables
frame_queue = Queue(maxsize=10)  # Limit on the queue size to manage memory
result_queue = Queue()
SOUND_FILE = 'welcome_soundRU.wav'
sound_played_event = threading.Event()  # Event to control sound playback

# Set the model to use for recognition
recognition_model = models[1]  # Change this index to use a different model, e.g., "ArcFace"

def play_sound():
    playsound(SOUND_FILE)

def analyze_frame():
    temp_path = '/tmp/temp_frame.jpg'  # Temporary path for saving frames

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            logging.debug("Analyzing frame with shape: %s", frame.shape)
            # Resize the frame for analysis
            frame_resized = cv2.resize(frame, (640, 480))
            cv2.imwrite(temp_path, frame_resized)
            
            try:
                # Perform analysis using DeepFace with a specific model
                analysis = DeepFace.analyze(img_path=temp_path, actions=['gender', 'age', 'emotion'], enforce_detection=False, detector_backend='opencv')
                
                logging.debug("Analysis results: %s", analysis)  # Отладочный вывод

                if not analysis:  # Check if analysis result is empty
                    result_queue.put(["No face detected"])
                    continue
                
                # Process results
                processed_results = process_results(analysis)
                logging.debug("Processed results: %s", processed_results)  # Отладочный вывод

                # Add results to the corresponding queues
                gender_queue.append(processed_results[0])
                age_queue.append(processed_results[1])
                emotion_queue.append(processed_results[2])
                
                # Get most common results
                most_common_gender = compute_most_common(gender_queue)
                most_common_age = compute_most_common(age_queue)
                most_common_emotion = compute_most_common(emotion_queue)
                
                # Обновляем глобальные результаты
                current_results = [most_common_gender, most_common_age, most_common_emotion]
                result_queue.put(current_results)

            except Exception as e:
                logging.error(f"Analysis error: {e}")  # Use logging for error output
                result_queue.put([f"Analysis error: {e}"])

            timer = threading.Timer(5.0, set_visualise_true)
            timer.start()

def process_results(analysis):
    gender_result = None
    age_result = None
    emotion_result = None
    
    for result in analysis:
        if 'gender' in result:
            gender = result['gender']
            max_gender = max(gender.items(), key=lambda x: x[1])
            gender_result = f"Gender: {max_gender[0]} ({max_gender[1]:.2f})"
        
        if 'age' in result:
            age = result['age']
            age_result = f"Age: {age}"
        
        if 'emotion' in result:
            emotions = result['emotion']
            max_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_result = f"Emotion: {max_emotion[0]} ({max_emotion[1]:.2f})"
    
    # Возвращаем список с результатами
    return [gender_result, age_result, emotion_result]

def compute_most_common(results):
    # Вычисление наиболее частого значения из списка
    if results:
        count = Counter(results)
        most_common = count.most_common(1)
        return most_common[0][0] if most_common else None
    return None

def create_black_background(width, height):
    return np.zeros((height, width, 3), dtype=np.uint8)

def main():
    global current_results  # Declare that current_results is a global variable

    current_results = [None, None, None]  # Initialize current_results

    # Open video file
    video_path = 'test_video_file_3.mp4'  # Replace with your video file path
    cap = cv2.VideoCapture(1)

    # Create a window and set it to resize mode
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    window_width = 640
    window_height = 480
    cv2.resizeWindow('Video', window_width, window_height)

    # Get video dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        cap.release()
        cv2.destroyAllWindows()
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
            # If the end of the video is reached, reset to the beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
            logging.debug("Results obtained from queue: %s", current_results)

        # Объединяем все результаты в одну строку
        display_text = '\n'.join(str(item) for item in current_results if item is not None)

        # Display text from the analysis results
        y0, dy = 30, 30  # Начальная координата y и шаг между строками
        if isStartVisualise:
            for i, line in enumerate(current_results):
                if line:  # Check if the line is not None
                    cv2.putText(background, line, (10, y0 + i * dy), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        # Display video
        cv2.imshow('Video', background)
        if cv2.waitKey(1) & 0xFF == 27:  # Esc key code - 27
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()