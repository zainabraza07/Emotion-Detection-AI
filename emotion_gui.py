import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageTk
import cv2

model = load_model(r'C:\Users\Al hamad\.vscode\Array\AssignmentDSA\.vscode\facial_emotion_detection\saved_models\emotion_model.h5')

emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

def predict_emotion(img_path=None):
    if img_path:
        img_array = preprocess_image(img_path)
    else:
        img_array = preprocess_image_from_camera()

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class] * 100 
    emotion = emotions[predicted_class]

    label.config(text=f"Emotion: {emotion} ({confidence:.2f}% confidence)")
   
    img = Image.open(img_path if img_path else "captured_image.jpg")
    img = img.resize((150, 150))  
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img  


def open_camera():
    cap = cv2.VideoCapture(0)  
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Press 'c' to capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):  
            cv2.imwrite("captured_image.jpg", frame)
            break

    cap.release()
    cv2.destroyAllWindows()
    predict_emotion()  
def preprocess_image_from_camera():
    img = cv2.imread("captured_image.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48)) 
    img = np.expand_dims(img, axis=-1)  
    img = np.expand_dims(img, axis=0)  
    img = img / 255.0 
    return img

window = tk.Tk()
window.title("Emotion Detection")

label = tk.Label(window, text="FACIAL EMOTION DETECTION", font=("Arial", 16))
label.pack(pady=20)

button_upload = tk.Button(window, text="Select Image", command=lambda: predict_emotion(filedialog.askopenfilename()), font=("Arial", 14))
button_upload.pack(pady=20)

button_camera = tk.Button(window, text="Open Camera", command=open_camera, font=("Arial", 14))
button_camera.pack(pady=20)

panel = tk.Label(window)
panel.pack(pady=20)

window.mainloop()
