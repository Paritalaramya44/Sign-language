"""
Enhanced train.py for ASL Alphabet Dataset (87,000 images)
Designed for high accuracy sign language recognition
"""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Paths
dataset_path = 'processed_asl_data/asl_keypoint.csv'
models_dir = 'models'
model_h5_path = f'{models_dir}/asl_model.hdf5'
model_tflite_path = f'{models_dir}/asl_model.tflite'

class ASLModelTrainer:
    def __init__(self, num_classes=27):
        self.num_classes = num_classes
        os.makedirs(models_dir, exist_ok=True)

    def create_enhanced_model(self, input_shape=(42,)):
        """Create enhanced model architecture for ASL recognition"""
        model = Sequential([
            # Input layer with L2 regularization
            Dense(512, activation='relu', input_shape=input_shape, 
                  kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),

            # Hidden layers with progressive size reduction
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.4),

            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),

            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),

            Dense(32, activation='relu'),
            Dropout(0.1),

            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])

        return model

    def load_data(self):
        """Load and preprocess ASL dataset"""
        print("[INFO] Loading ASL dataset...")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run asl_data_processor.py first!")

        # Load data
        data = np.loadtxt(dataset_path, delimiter=',', dtype='float32')
        X = data[:, 1:43]  # Features (42 landmarks: 21 points × 2 coordinates)
        y = data[:, 0].astype('int32')  # Labels

        print(f"[INFO] Dataset shape: {X.shape}")
        print(f"[INFO] Labels shape: {y.shape}")
        print(f"[INFO] Number of classes: {len(np.unique(y))}")
        print(f"[INFO] Feature range: [{X.min():.3f}, {X.max():.3f}]")

        # Verify data quality
        if X.shape[1] != 42:
            raise ValueError(f"Expected 42 features, got {X.shape[1]}")

        # Check for NaN values
        if np.isnan(X).any() or np.isnan(y).any():
            print("[WARNING] Found NaN values in dataset!")
            # Remove NaN samples
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            print(f"[INFO] After removing NaN: {X.shape}")

        return X, y

    def train_model(self, use_cross_validation=False):
        """Train the enhanced ASL recognition model"""
        # Load data
        X, y = self.load_data()

        if use_cross_validation:
            return self._train_with_cv(X, y)
        else:
            return self._train_single_model(X, y)

    def _train_single_model(self, X, y):
        """Train a single model with train/val/test split"""
        # Split data: 70% train, 15% validation, 15% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
        )

        print(f"[INFO] Training set: {X_train.shape}")
        print(f"[INFO] Validation set: {X_val.shape}")
        print(f"[INFO] Test set: {X_test.shape}")

        # Create and compile model
        model = self.create_enhanced_model()

        # Use AdamW optimizer with weight decay
        optimizer = AdamW(
            learning_rate=0.001,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999
        )

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )

        # Model summary
        print("\n[INFO] Model Architecture:")
        model.summary()

        # Advanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                model_h5_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        # Train model
        print("\n[INFO] Starting training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=128,
            callbacks=callbacks,
            verbose=1
        )

        # Load best model
        model = tf.keras.models.load_model(model_h5_path)

        # Final evaluation
        print("\n[INFO] Final Evaluation:")

        # Training accuracy
        train_loss, train_acc, train_top_k = model.evaluate(X_train, y_train, verbose=0)
        print(f"Training Accuracy: {train_acc:.4f}")

        # Validation accuracy
        val_loss, val_acc, val_top_k = model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation Accuracy: {val_acc:.4f}")

        # Test accuracy
        test_loss, test_acc, test_top_k = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Top-K Accuracy: {test_top_k:.4f}")

        # Detailed classification report
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        print("\n[INFO] Classification Report:")
        print(classification_report(y_test, y_pred))

        # Save performance metrics
        metrics = {
            'train_accuracy': float(train_acc),
            'validation_accuracy': float(val_acc),
            'test_accuracy': float(test_acc),
            'test_top_k_accuracy': float(test_top_k),
            'model_params': model.count_params(),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }

        with open(f'{models_dir}/training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # Convert to TensorFlow Lite
        self._convert_to_tflite(model)

        return model, history, metrics

    def _convert_to_tflite(self, model):
        """Convert trained model to TensorFlow Lite format"""
        print("\n[INFO] Converting to TensorFlow Lite...")

        # Convert model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Optimization (optional)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        # Save TFLite model
        with open(model_tflite_path, 'wb') as f:
            f.write(tflite_model)

        print(f"[INFO] TFLite model saved to {model_tflite_path}")
        print(f"[INFO] Model size: {len(tflite_model) / 1024:.2f} KB")

    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Loss plot
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f'{models_dir}/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main training function"""
    print("="*60)
    print("ASL Alphabet Dataset - Enhanced Model Training")
    print("="*60)

    # Initialize trainer
    trainer = ASLModelTrainer(num_classes=27)

    try:
        # Train model
        model, history, metrics = trainer.train_model()

        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Model saved to: {model_h5_path}")
        print(f"TFLite model saved to: {model_tflite_path}")
        print("\nYou can now use the trained model with your existing app.py!")

        # Plot training history
        trainer.plot_training_history(history)

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        print("\nMake sure you have:")
        print("1. Processed the ASL dataset using asl_data_processor.py")
        print("2. Installed all required dependencies")

if __name__ == "__main__":
    main()
