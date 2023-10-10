import pickle
import cv2
import mediapipe as mp
import numpy as np

loaded_dict = pickle.load(open('./model.p', 'rb'))
hand_model = loaded_dict['model']

video_capture = cv2.VideoCapture(0)

media_hands = mp.solutions.hands
media_drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

hand_detector = media_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

#1.create dict
gesture_labels = {0: 'A', 1: 'B', 2: 'L'}
prob = None
while True:
    predicted_label = "Not confident"
    temp_data = []
    x_coordinates = []
    y_coordinates = []

    is_captured, video_frame = video_capture.read()

    frame_height, frame_width, _ = video_frame.shape

    frame_in_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)

    detection_results = hand_detector.process(frame_in_rgb)
    if detection_results.multi_hand_landmarks:
        #Draw hands
        for landmark_set in detection_results.multi_hand_landmarks:
            media_drawing.draw_landmarks(
                video_frame,
                landmark_set,
                media_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style())
            
        
        for landmark_set in detection_results.multi_hand_landmarks:
            #Extract x and y
            for i in range(len(landmark_set.landmark)):
                x_val = landmark_set.landmark[i].x
                y_val = landmark_set.landmark[i].y

                x_coordinates.append(x_val)
                y_coordinates.append(y_val)
            #Normalize x and y
            for i in range(len(landmark_set.landmark)):
                x_val = landmark_set.landmark[i].x
                y_val = landmark_set.landmark[i].y
                temp_data.append(x_val - min(x_coordinates))
                temp_data.append(y_val - min(y_coordinates))

        top_left_x = int(min(x_coordinates) * frame_width) - 10
        top_left_y = int(min(y_coordinates) * frame_height) - 10

        gesture_prediction = hand_model.predict([np.asarray(temp_data)])
        prob = hand_model.predict_proba([np.asarray(temp_data)])
        
        #modify confidence.
        if np.max(prob) >= 0.8:
            predicted_label = gesture_labels[int(gesture_prediction[0])]

        cv2.putText(video_frame, predicted_label, (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
        if np.max(prob):
            cv2.putText(video_frame, str(np.max(prob)), (top_left_x, top_left_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    cv2.imshow('text', video_frame)
    cv2.waitKey(1)

video_capture.release()
cv2.destroyAllWindows()
