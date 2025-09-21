import os
import cv2 as cv
import mediapipe as mp
import csv

def process_image(image_path):
    """Processes a single image to extract hand landmarks."""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    image = cv.imread(image_path)
    if image is None:
        return None
        
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y])
        return landmarks
    return None

def create_landmark_dataset(dataset_path, output_csv_path):
    """Creates a landmark dataset from a directory of images."""
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        for label_dir in sorted(os.listdir(dataset_path)):
            label_path = os.path.join(dataset_path, label_dir)
            if os.path.isdir(label_path):
                label_index = ord(label_dir) - ord('A')
                for image_file in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_file)
                    landmarks = process_image(image_path)
                    if landmarks:
                        row = [label_index] + landmarks
                        csv_writer.writerow(row)
                        print(f"Processed {image_path}")

if __name__ == '__main__':
    train_dataset_path = 'asl_dataset/asl_alphabet_train/asl_alphabet_train'
    output_train_csv_path = 'slr/model/new_keypoint.csv'
    
    print("Processing training dataset...")
    create_landmark_dataset(train_dataset_path, output_train_csv_path)
    
    print("Dataset processing complete.")
