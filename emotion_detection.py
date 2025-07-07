import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Step 1: Load the CSV file into a DataFrame
df = pd.read_csv(r'C:\Users\Al hamad\Downloads\archive (1)\fer2013.csv')  # Correct file path

# Check the first few rows to ensure the data is loaded correctly
print(df.head())

# Step 2: Convert the 'pixels' column into a 2D array of images (pixels)
X = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))  # Split the pixel string into numbers
X = np.array(X.tolist())  # Convert to a 2D NumPy array

# Step 3: Normalize the pixel values to range [0, 1]
X = X / 255.0

# Step 4: Reshape the images to 48x48 pixels (grayscale)
X = X.reshape(-1, 48, 48, 1)  # Shape is (number of images, 48x48 pixels, 1 color channel)

# Step 5: One-hot encode the labels (emotions)
y = df['emotion'].values
y = to_categorical(y, num_classes=7)  # 7 classes, adjust if needed

# Step 6: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data loaded and prepared successfully!")

# Step 7: Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # This helps avoid overfitting
    Dense(7, activation='softmax')  # 7 output classes (emotions)
])

# Step 8: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 9: Train the model and capture training history (this should only run once)
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Step 10: Save the trained model once after training
model.save(r'C:\Users\Al hamad\.vscode\Array\AssignmentDSA\.vscode\facial_emotion_detection\saved_models\emotion_model.h5')

# Step 11: Print the training and validation loss and accuracy for each epoch
print("Training History:")
print(f"Training Loss: {history.history['loss']}")
print(f"Training Accuracy: {history.history['accuracy']}")
print(f"Validation Loss: {history.history['val_loss']}")
print(f"Validation Accuracy: {history.history['val_accuracy']}")
