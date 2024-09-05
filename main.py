import cv2
import threading
from deepface import DeepFace
from queue import Queue
import numpy as np

# Глобальная переменная для хранения текущего кадра
frame_queue = Queue(maxsize=10)  # Ограничение на размер очереди для управления памятью
result_queue = Queue()
current_results = []

def analyze_frame():
    temp_path = '/tmp/temp_frame.jpg'  # Измените путь в зависимости от вашей ОС
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Уменьшение разрешения для анализа
            frame_resized = cv2.resize(frame, (640, 480))
            cv2.imwrite(temp_path, frame_resized)
            
            try:
                # Выполнение анализа с использованием DeepFace
                analysis = DeepFace.analyze(img_path=temp_path, actions=['gender', 'age'], enforce_detection=False)
                
                # Фильтрация и форматирование результата анализа
                gender_result = []
                age_result = None
                for result in analysis:
                    if 'gender' in result:
                        gender = result['gender']
                        # Получение пола с максимальной уверенностью
                        max_gender = max(gender.items(), key=lambda x: x[1])
                        gender_result.append(f"Gender: {max_gender[0]} ({max_gender[1]:.2f})")
                    
                    if 'age' in result:
                        age = result['age']
                        age_result = f"Age: {age}"
                
                # Сохранение только наиболее уверенного пола и возраста
                filtered_results = []
                if gender_result:
                    filtered_results.append(max(gender_result, key=lambda x: float(x.split('(')[-1].split(')')[0])))
                if age_result:
                    filtered_results.append(age_result)
                
                result_queue.put(filtered_results)
            except Exception as e:
                result_queue.put([f"Ошибка анализа: {e}"])

def create_black_background(width, height):
    return np.zeros((height, width, 3), dtype=np.uint8)

def main():
    global current_results  # Объявляем, что current_results - глобальная переменная
    cap = cv2.VideoCapture(0)

    # Создание окна и установка его в режим изменения размеров
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    window_width = 640
    window_height = 480
    cv2.resizeWindow('Video', window_width, window_height)

    # Получение размеров камеры
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return
    frame_height, frame_width = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (0, 255, 0)
    font_thickness = 2

    # Создание и запуск потока для анализа
    analysis_thread = threading.Thread(target=analyze_frame, daemon=True)
    analysis_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Изменение размера кадра для отображения с сохранением соотношения сторон
        if frame_width > frame_height:
            new_width = window_width
            new_height = int((window_width / frame_width) * frame_height)
        else:
            new_height = window_height
            new_width = int((window_height / frame_height) * frame_width)

        frame_resized = cv2.resize(frame, (new_width, new_height))
        # Создание черного фона для центрирования изображения
        background = create_black_background(window_width, window_height)
        y_offset = (window_height - new_height) // 2
        x_offset = (window_width - new_width) // 2
        background[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = frame_resized

        # Попытка добавления кадра в очередь для анализа
        if frame_queue.qsize() < 10:  # Проверка размера очереди
            frame_queue.put(frame)

        # Получение результатов анализа из очереди, если они есть
        if not result_queue.empty():
            current_results = result_queue.get()

        # Отображение текста из результатов анализа
        y0, dy = 30, 30
        with result_queue.mutex:
            for i, text in enumerate(current_results):
                cv2.putText(background, text, (10, y0 + i * dy), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        # Отображение видео
        cv2.imshow('Video', background)
        if cv2.waitKey(1) & 0xFF == 27:  # Код клавиши 'Esc' - 27
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()