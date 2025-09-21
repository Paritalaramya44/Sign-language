print("INFO: Initializing System")
import copy
import csv
import os
import datetime
import time

import pyautogui
import cv2 as cv
import mediapipe as mp
from dotenv import load_dotenv

from slr.model.classifier import KeyPointClassifier

from slr.utils.args import get_args
from slr.utils.cvfpscalc import CvFpsCalc
from slr.utils.landmarks import draw_landmarks

from slr.utils.draw_debug import get_result_image
from slr.utils.draw_debug import get_fps_log_image
from slr.utils.draw_debug import draw_bounding_rect
from slr.utils.draw_debug import draw_hand_label
from slr.utils.draw_debug import show_fps_log
from slr.utils.draw_debug import show_result

from slr.utils.pre_process import calc_bounding_rect
from slr.utils.pre_process import calc_landmark_list
from slr.utils.pre_process import pre_process_landmark

from slr.utils.logging import log_keypoints
from slr.utils.logging import get_dict_form_list
from slr.utils.logging import get_mode

# Text-to-Speech import (optional)
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("WARNING: pyttsx3 not installed. Voice feature will be disabled.")

class SentenceBuilder:
    def __init__(self):
        self.sentence = ""
        self.last_prediction = ""
        self.prediction_time = 0
        self.stability_threshold = 1.0  # seconds to wait for stable prediction
        self.last_add_time = 0
        self.min_add_interval = 0.5  # minimum seconds between adding characters
        
        # Initialize TTS if available
        if TTS_AVAILABLE:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)  # Speed of speech
            self.tts_engine.setProperty('volume', 0.7)  # Volume level
        else:
            self.tts_engine = None
    
    def add_prediction(self, prediction):
        current_time = time.time()
        
        # If same prediction as before, check stability
        if prediction == self.last_prediction and prediction != "":
            if current_time - self.prediction_time >= self.stability_threshold:
                if current_time - self.last_add_time >= self.min_add_interval:
                    self.add_character(prediction)
                    self.last_add_time = current_time
        else:
            # New prediction, reset timer
            self.last_prediction = prediction
            self.prediction_time = current_time
    
    def add_character(self, character):
        if character and character != "":
            self.sentence += character
            print(f"Added character: {character}")
            print(f"Current sentence: {self.sentence}")
    
    def add_space(self):
        if self.sentence and not self.sentence.endswith(" "):
            self.sentence += " "
            print("Added space")
    
    def delete_last_character(self):
        if self.sentence:
            self.sentence = self.sentence[:-1]
            print("Deleted last character")
            print(f"Current sentence: {self.sentence}")
    
    def clear_sentence(self):
        self.sentence = ""
        print("Sentence cleared")
    
    def speak_sentence(self):
        if self.tts_engine and self.sentence:
            print(f"Speaking: {self.sentence}")
            self.tts_engine.say(self.sentence)
            self.tts_engine.runAndWait()
        elif not TTS_AVAILABLE:
            print("TTS not available. Please install pyttsx3: pip install pyttsx3")
        else:
            print("No sentence to speak")
    
    def get_sentence(self):
        return self.sentence

def draw_sentence_on_image(image, sentence, position=(50, 50)):
    """Draw the current sentence on the image"""
    # Create a semi-transparent overlay for better text visibility
    overlay = image.copy()
    cv.rectangle(overlay, (position[0]-10, position[1]-30), 
                (len(sentence)*15 + position[0] + 20, position[1]+10), 
                (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Draw the sentence text
    cv.putText(image, sentence, position, cv.FONT_HERSHEY_SIMPLEX, 
               0.8, (255, 255, 255), 2, cv.LINE_AA)
    return image

def draw_instructions(image):
    """Draw control instructions on the image"""
    instructions = [
        "Controls:",
        "ESC - Exit",
        "SPACE - Add space",
        "BACKSPACE - Delete last char", 
        "DELETE - Clear sentence",
        "ENTER - Speak sentence",
        "9 - Screenshot"
    ]
    
    start_y = 400
    for i, instruction in enumerate(instructions):
        cv.putText(image, instruction, (750, start_y + i*25), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    return image

def main():
    #: -
    #: Getting all arguments
    load_dotenv()
    args = get_args()

    keypoint_file = "slr/model/keypoint.csv"
    counter_obj = get_dict_form_list(keypoint_file)

    #: cv Capture
    CAP_DEVICE = args.device
    CAP_WIDTH = args.width
    CAP_HEIGHT = args.height

    #: mp Hands
    USE_STATIC_IMAGE_MODE = True
    MAX_NUM_HANDS = args.max_num_hands
    MIN_DETECTION_CONFIDENCE = args.min_detection_confidence
    MIN_TRACKING_CONFIDENCE = args.min_tracking_confidence

    #: Drawing Rectangle
    USE_BRECT = args.use_brect
    MODE = args.mode
    DEBUG = int(os.environ.get("DEBUG", "0")) == 1
    CAP_DEVICE = 0

    # Initialize sentence builder
    sentence_builder = SentenceBuilder()

    print("INFO: System initialization Successful")
    print("INFO: Opening Camera")

    #: -
    #: Capturing image
    cap = cv.VideoCapture(CAP_DEVICE)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    
    background_image = cv.imread("resources/background_prediction.png")

    #: Background Image
    background_image = cv.imread("resources/background.png")

    #: -
    #: Setup hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=USE_STATIC_IMAGE_MODE,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )

    #: -
    #: Load Model
    keypoint_classifier = KeyPointClassifier()

    #: Loading labels
    keypoint_labels_file = "slr/model/label.csv"
    with open(keypoint_labels_file, encoding="utf-8-sig") as f:
        key_points = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in key_points]

    #: -
    #: FPS Measurement
    cv_fps = CvFpsCalc(buffer_len=10)
    print("INFO: System is up & running")
    print("\nControls:")
    print("ESC - Exit")
    print("SPACE - Add space to sentence")
    print("BACKSPACE - Delete last character")
    print("DELETE - Clear entire sentence")
    print("ENTER - Speak current sentence (if TTS available)")
    print("9 - Take screenshot")
    
    #: -
    #: Main Loop Start Here...
    while True:
        #: FPS of open cv frame or window
        fps = cv_fps.get()

        #: -
        #: Setup Quit key for program
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            print("INFO: Exiting...")
            break
        elif key == 57:  # 9 key
            name = datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
            myScreenshot = pyautogui.screenshot()
            myScreenshot.save(f'ss/{name}.png')
            print(f"Screenshot saved as ss/{name}.png")
        elif key == 32:  # SPACE key
            sentence_builder.add_space()
        elif key == 8:  # BACKSPACE key
            sentence_builder.delete_last_character()
        elif key == 127 or key == 46:  # DELETE key
            sentence_builder.clear_sentence()
        elif key == 13:  # ENTER key
            sentence_builder.speak_sentence()

        #: -
        #: Camera capture
        success, image = cap.read()
        if not success:
            continue
        
        image = cv.resize(image, (CAP_WIDTH, CAP_HEIGHT))
        
        #: Flip Image for mirror display
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        result_image = get_result_image()
        fps_log_image = get_fps_log_image()

        #: Converting to RBG from BGR
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)  #: Hand's landmarks
        image.flags.writeable = True

        #: -
        #: DEBUG - Showing Debug info
        if DEBUG:
            MODE = get_mode(key, MODE)
            fps_log_image = show_fps_log(fps_log_image, fps)

        #: Initialize prediction variables
        hand_sign_text = ""

        #: -
        #: Start Detection
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                #: Calculate  BoundingBox
                use_brect = True
                brect = calc_bounding_rect(debug_image, hand_landmarks)

                #: Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                #: Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                #: -
                #: Checking if in Prediction Mode or in Logging Mode
                if MODE == 0:  #: Prediction Mode / Normal mode
                    #: Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                    if hand_sign_id == 25:
                        hand_sign_text = ""
                    else:
                        hand_sign_text = keypoint_classifier_labels[hand_sign_id]

                    #: Add prediction to sentence builder
                    sentence_builder.add_prediction(hand_sign_text)

                    #: Showing Result
                    result_image = show_result(result_image, handedness, hand_sign_text)

                elif MODE == 1:  #: Logging Mode
                    log_keypoints(key, pre_processed_landmark_list, counter_obj, data_limit=1000)

                #: -
                #: Drawing debug info
                debug_image = draw_bounding_rect(debug_image, use_brect, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_hand_label(debug_image, brect, handedness)

        #: -
        #: Set main video footage on Background
        background_image = cv.imread("resources/background.png")
        background_image[170:170 + 480, 50:50 + 640] = debug_image
        background_image[240:240 + 127, 731:731 + 299] = result_image
        background_image[678:678 + 30, 118:118 + 640] = fps_log_image

        #: Draw the current sentence on the background
        current_sentence = sentence_builder.get_sentence()
        background_image = draw_sentence_on_image(background_image, 
                                                f"Sentence: {current_sentence}", 
                                                (50, 120))
        
        #: Draw current prediction
        background_image = draw_sentence_on_image(background_image, 
                                                f"Current: {hand_sign_text}", 
                                                (50, 150))
        
        #: Draw instructions
        background_image = draw_instructions(background_image)
        
        cv.imshow("Sign Language Recognition", background_image)

    cap.release()
    cv.destroyAllWindows()

    print("INFO: Bye")


if __name__ == "__main__":
    main()