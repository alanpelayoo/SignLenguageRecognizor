#Imports
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

#Create objects
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#Instance of Hands for hands tracking
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

#path to image
img_path = './data_q/0/0.jpg'

#read image
img = cv2.imread(img_path)

#convert image to RGB colorspace 
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#process hands tracking
results = hands.process(img_rgb)

#multi_hand_landmarks list contains an element for each hand, lenght = num of hands in img
# x_ = []
# y_ = []
# data_aux = []
# for hand_landmarks in results.multi_hand_landmarks:
#     for i in range(len(hand_landmarks.landmark)):
#         x = hand_landmarks.landmark[i].x
#         y = hand_landmarks.landmark[i].y

#         x_.append(x)
#         y_.append(y)

#     for i in range(len(hand_landmarks.landmark)):
#         x = hand_landmarks.landmark[i].x
#         y = hand_landmarks.landmark[i].y
#         data_aux.append(x - min(x_))

# index = x_.index(min(x_))
# print(x_)
# print(min(x_))
# print(index)

# print("data aux")
# print(data_aux)

# print(data_aux.index(0))

if results.multi_hand_landmarks:
    #draw
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            img_rgb,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

plt.figure()
plt.imshow(img_rgb)
plt.show()