#
# Face Recognition using Haar Cascade Detection and CNN Features (Python)
#
# This is the ADVANCED version. It is much more accurate than the LBP script.
#
# 1.  Detection:     Uses OpenCV's Haar Cascade detector.
# 2.  Features:      Uses a pre-trained ResNet-50 CNN (Deep Learning).
# 3.  Classification:Trains an SVM on those deep features.
#
# It also runs on your live webcam and will ask if you want to re-train.
#
# *** UPDATED: Now shows best guess. Box is RED if < 50% confidence. ***
#

import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# --- Settings ---
FACE_DATABASE_DIR = 'FaceDatabase'
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
MODEL_CLASSIFIER_PATH = 'cnn_classifier.pkl' # "Saved Brain" for CNN
MODEL_ENCODER_PATH = 'labels_encoder.pkl'    # "Saved Key" (shared)
CNN_INPUT_SIZE = (224, 224) # ResNet-50 input size
# We will show the name, but change color based on this threshold:
CONFIDENCE_THRESHOLD = 0.50 # 50%

# --- Step 1: Initialize Face Detector ---
print(f"Loading Haar cascade detector from {HAAR_CASCADE_PATH}...")
face_detector = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
if face_detector.empty():
    print("Error: Could not load Haar cascade. Check file path.")
    exit()

# --- Step 2: Load Pre-trained CNN (ResNet-50) ---
def load_cnn_extractor():
    print("Loading pre-trained CNN (ResNet-50) as feature extractor...")
    # Load ResNet-50, pre-trained on ImageNet
    # include_top=False means we don't include the final classification layer
    # pooling='avg' adds a global average pooling layer to get a 1D feature vector
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*CNN_INPUT_SIZE, 3), pooling='avg')
        
    # We use this model as-is for feature extraction
    model = Model(inputs=base_model.input, outputs=base_model.output)
    print("CNN loaded successfully.")
    return model

# --- Step 3: Function to get all image paths and labels ---
def get_image_data(database_path):
    print("Loading face database...")
    image_paths = []
    labels = []

    if not os.path.exists(database_path):
        print(f"Error: Directory '{database_path}' not found.")
        print("Please create it and add subfolders for each person.")
        return None, None

    for person_name in os.listdir(database_path):
        person_dir = os.path.join(database_path, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(image_path)
                    labels.append(person_name)
                
    print(f"Found {len(image_paths)} images for {len(set(labels))} people.")
    return image_paths, labels

# --- Step 4: Extract Deep Features from all Training Images ---
def train_model():
    image_paths, labels = get_image_data(FACE_DATABASE_DIR)
    if not image_paths:
        return None, None
        
    cnn_extractor = load_cnn_extractor()
    if cnn_extractor is None:
        return None, None

    print("Extracting CNN features from training database... (This may take a while)")
    training_features = []
    training_labels = []

    for i, (image_path, label) in enumerate(zip(image_paths, labels)):
        print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        # Read the image (CNNs need color)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read {image_path}. Skipping.")
            continue
            
        # Detect the face
        # We convert to gray *only* for the detector
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bboxes = face_detector.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(bboxes) > 0:
            # We found a face! Use the first one.
            (x, y, w, h) = bboxes[0]
            
            # Crop the face from the *original color image*
            face = img[y:y+h, x:x+w]
            
            # Resize to the CNN's required input size
            face_resized = cv2.resize(face, CNN_INPUT_SIZE)
            
            # Prepare image for ResNet-50
            # 1. Convert to array
            # 2. Add a "batch" dimension (e.g., (224, 224, 3) -> (1, 224, 224, 3))
            # 3. Pre-process (normalizes pixel values for the model)
            face_array = img_to_array(face_resized)
            face_expanded = np.expand_dims(face_array, axis=0)
            face_preprocessed = preprocess_input(face_expanded)
            
            # Extract features
            # This returns a 1x2048 vector
            features = cnn_extractor.predict(face_preprocessed, verbose=0)
            
            training_features.append(features.flatten()) # flatten to 1D
            training_labels.append(label)
        else:
            print(f"Warning: No face detected in {image_path}. Skipping.")
            
    if not training_features:
        print("Error: No features were extracted. Check your 'FaceDatabase' images.")
        return None, None

    print(f"Feature extraction complete. Extracted {len(training_features)} features.")

    # --- Step 5: Train a Classifier ---
    print("Training the face recognizer (SVM on deep features)...")
    
    # Encode string labels
    label_encoder = LabelEncoder()
    training_labels_encoded = label_encoder.fit_transform(training_labels)
    
    # Train the SVM
    face_classifier = SVC(kernel='linear', C=1.0, probability=True)
    face_classifier.fit(training_features, training_labels_encoded)
    
    print("Classifier trained. Saving models...")
    
    # --- Save the trained models ---
    with open(MODEL_CLASSIFIER_PATH, 'wb') as f:
        pickle.dump(face_classifier, f)
    with open(MODEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
        
    print(f"Models saved to {MODEL_CLASSIFIER_PATH} and {MODEL_ENCODER_PATH}")
    return face_classifier, label_encoder, cnn_extractor

# --- Step 6: Test the Recognizer on a Live Webcam Feed ---
def recognize_faces_live():
    cnn_extractor = None
    force_retrain = False

    # --- Check if user wants to re-train ---
    models_exist = os.path.exists(MODEL_CLASSIFIER_PATH) and os.path.exists(MODEL_ENCODER_PATH)
    
    if models_exist:
        while True:
            # Ask user if they want to re-train
            user_input = input("Saved models found. Do you want to re-train? (y/n): ").lower().strip()
            if user_input == 'y':
                force_retrain = True
                break
            elif user_input == 'n':
                force_retrain = False
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    
    # Load or train models
    if force_retrain or not models_exist:
        if force_retrain:
            print("Re-training as requested...")
        else:
            print("Models not found. Training new models...")
        classifier, encoder, cnn_extractor = train_model()
        if classifier is None:
            return
    else:
        print("Loading pre-trained models...")
        with open(MODEL_CLASSIFIER_PATH, 'rb') as f:
            classifier = pickle.load(f)
        with open(MODEL_ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
            
    # Load the CNN extractor if it wasn't loaded during training
    if cnn_extractor is None:
        cnn_extractor = load_cnn_extractor()

    # --- Initialize Webcam ---
    print("Starting webcam... Press 'q' to quit.")
    cap = cv2.VideoCapture(0) # 0 is the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        
        # Flip the frame horizontally (for a natural mirror effect)
        frame = cv2.flip(frame, 1)
        
        # We need two versions of the frame:
        # 1. Grayscale for the *detector*
        # 2. Color for the *feature extractor*
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect all faces
        test_bboxes = face_detector.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5)

        if len(test_bboxes) > 0:
            for (x, y, w, h) in test_bboxes:
                # Crop the face from the *original color frame*
                test_face_color = frame[y:y+h, x:x+w]
                
                # Resize and preprocess for CNN
                test_face_resized = cv2.resize(test_face_color, CNN_INPUT_SIZE)
                face_array = img_to_array(test_face_resized)
                face_expanded = np.expand_dims(face_array, axis=0)
                face_preprocessed = preprocess_input(face_expanded)
            
                # Extract deep features
                test_features = cnn_extractor.predict(face_preprocessed, verbose=0)
                
                # Predict the person's identity
                probabilities = classifier.predict_proba(test_features)[0]
                predicted_index = np.argmax(probabilities)
                confidence = probabilities[predicted_index] 
                
                # Get the label name (we always show the best guess)
                predicted_label_name = encoder.inverse_transform([predicted_index])[0]
                annotation = f"{predicted_label_name} ({confidence*100:.1f}%)"

                # ******** THIS IS THE NEW LOGIC ********
                # Change color based on confidence
                if confidence > CONFIDENCE_THRESHOLD:
                    color = (0, 255, 0) # Green for > 50%
                else:
                    color = (0, 0, 255) # Red for < 50%
                
                # Draw rectangle and text on the original color frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, annotation, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
        else:
            # No faces detected in this frame
            pass

        # Display the final result
        cv2.imshow("Live Face Recognition (CNN) - Press 'q' to quit", frame)
        
        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed stopped.")

# --- Main execution ---
if __name__ == "__main__":
    # Suppress TensorFlow informational messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
    tf.get_logger().setLevel('ERROR')
    recognize_faces_live()