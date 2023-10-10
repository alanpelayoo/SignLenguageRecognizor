import cv2
import os

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera!")
    exit()

count = 0  # Number of images captured so far
total_images = 5  # Total images to capture

print("Press 'q' to capture an image.")

while count < total_images:
    ret, frame = cap.read()
    
    # Check if frame is successfully grabbed
    if not ret:
        print("Failed to grab frame!")
        break
    
    # Display the live video feed
    cv2.imshow('Capture Images', frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        img_name = f"image_{count+1}.jpg"
        img_path = os.path.join("samples",img_name)
        cv2.imwrite(img_path, frame)
        print(f"Captured {img_name}")
        count += 1

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()