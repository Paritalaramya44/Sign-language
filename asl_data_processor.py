"""
ASL Data Processor for converting ASL Alphabet Dataset to MediaPipe landmarks
"""
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm

class ASLDataProcessor:
    def __init__(self, dataset_path, output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Class mapping matching your current project structure
        self.class_mapping = {
            'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
            'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11,
            'N': 12, 'O': 13, 'P': 14, 'Q': 15, 'R': 16, 'S': 17,
            'T': 18, 'U': 19, 'V': 20, 'W': 21, 'X': 22, 'Y': 23,
            'SPACE': 24, 'DELETE': 25, 'NOTHING': 26
        }

    def extract_landmarks(self, image_path):
        """Extract MediaPipe landmarks from image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                # Extract normalized coordinates (21 landmarks × 2 coordinates = 42 features)
                landmark_list = []
                for landmark in landmarks.landmark:
                    landmark_list.extend([landmark.x, landmark.y])
                return landmark_list
            return None
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def normalize_landmarks(self, landmark_list):
        """Normalize landmarks relative to wrist (same as your current preprocessing)"""
        if len(landmark_list) != 42:
            return None

        # Convert to numpy array and reshape
        landmarks = np.array(landmark_list).reshape(21, 2)

        # Get wrist position (landmark 0)
        wrist_x, wrist_y = landmarks[0]

        # Make relative to wrist
        landmarks[:, 0] -= wrist_x
        landmarks[:, 1] -= wrist_y

        # Normalize to [-1, 1] range
        max_val = np.max(np.abs(landmarks))
        if max_val > 0:
            landmarks = landmarks / max_val

        return landmarks.flatten()

    def process_dataset(self, max_samples_per_class=None):
        """Process entire ASL dataset"""
        os.makedirs(self.output_path, exist_ok=True)

        data = []
        labels = []
        processed_count = {}

        print("Starting ASL dataset processing...")

        for class_name in os.listdir(self.dataset_path):
            if class_name not in self.class_mapping:
                continue

            class_path = os.path.join(self.dataset_path, class_name)
            if not os.path.isdir(class_path):
                continue

            class_label = self.class_mapping[class_name]
            image_files = os.listdir(class_path)

            # Limit samples per class if specified
            if max_samples_per_class:
                image_files = image_files[:max_samples_per_class]

            print(f"Processing class: {class_name} (Label: {class_label}) - {len(image_files)} images")

            successful_extractions = 0

            for image_file in tqdm(image_files, desc=f"Processing {class_name}"):
                if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                image_path = os.path.join(class_path, image_file)
                landmarks = self.extract_landmarks(image_path)

                if landmarks:
                    # Normalize landmarks (same preprocessing as your current project)
                    normalized_landmarks = self.normalize_landmarks(landmarks)
                    if normalized_landmarks is not None:
                        data.append(normalized_landmarks)
                        labels.append(class_label)
                        successful_extractions += 1

            processed_count[class_name] = successful_extractions
            print(f"Successfully processed {successful_extractions} images for class {class_name}")

        return np.array(data), np.array(labels), processed_count

    def save_processed_data(self, X, y, processed_count):
        """Save processed data in your existing format"""
        print(f"Saving {len(X)} processed samples...")

        # Combine labels and features (same format as your keypoint.csv)
        combined_data = np.column_stack([y, X])

        # Save to CSV
        output_csv = os.path.join(self.output_path, 'asl_keypoint.csv')
        np.savetxt(output_csv, combined_data, delimiter=',', fmt='%.8f')

        # Create counter.json
        counter_dict = {str(label): count for label, count in processed_count.items() 
                       if self.class_mapping.get(label) is not None}
        counter_path = os.path.join(self.output_path, 'asl_counter.json')
        with open(counter_path, 'w') as f:
            json.dump(counter_dict, f, indent=2)

        # Create labels file
        labels_list = list(self.class_mapping.keys())
        labels_path = os.path.join(self.output_path, 'asl_labels.csv')
        with open(labels_path, 'w') as f:
            f.write('\n'.join(labels_list))

        print(f"Data saved successfully!")
        print(f"CSV file: {output_csv}")
        print(f"Counter file: {counter_path}")
        print(f"Labels file: {labels_path}")

        return output_csv

# Usage example
def main():
    # Configure paths
    dataset_path = 'asl_dataset/asl_alphabet_train/asl_alphabet_train'  # Path to extracted ASL dataset
    output_path = 'processed_asl_data'               # Output directory

    # Initialize processor
    processor = ASLDataProcessor(dataset_path, output_path)

    # Process dataset (limit to 1000 samples per class for testing)
    X, y, processed_count = processor.process_dataset(max_samples_per_class=1000)

    if len(X) > 0:
        # Save processed data
        csv_path = processor.save_processed_data(X, y, processed_count)

        print(f"\nProcessing Summary:")
        print(f"Total samples: {len(X)}")
        print(f"Feature dimensions: {X.shape[1]}")
        print(f"Number of classes: {len(np.unique(y))}")
        print(f"Class distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for class_id, count in zip(unique, counts):
            class_name = [k for k, v in processor.class_mapping.items() if v == class_id][0]
            print(f"  {class_name} (ID {class_id}): {count} samples")
    else:
        print("No samples were successfully processed!")

if __name__ == "__main__":
    main()
