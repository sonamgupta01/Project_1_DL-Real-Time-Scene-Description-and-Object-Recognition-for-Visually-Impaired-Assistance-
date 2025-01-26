import os
import sys
import cv2
import pytesseract
import time
import traceback
import threading

import speech_recognition as sr
import pyttsx3


from PIL import Image
import torch
from torch.nn import functional as F

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image as KivyImage
from kivy.uix.scrollview import ScrollView
from kivy.graphics.texture import Texture
from kivy.core.window import Window

from transformers import BlipProcessor, BlipForConditionalGeneration
from torchvision import transforms as trn
from transformers import DetrFeatureExtractor, DetrForObjectDetection

Window.clearcolor = (0,0,0,1)

def log_error(msg):
    print(msg, file=sys.stderr)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = None
blip_model = None
places_model = None
categories = []
centre_crop = None
detr_feature_extractor = None
detr_model = None

def load_places365():
    global places_model, categories, centre_crop
    arch = 'resnet18'
    model_file = f'whole_{arch}_places365.pth.tar'
    if not os.path.exists(model_file):
        log_error("Places365 model not found. Attempting to download...")
        torch.hub.download_url_to_file(f'http://places2.csail.mit.edu/models_places365/{model_file}', model_file)

    print("Loading Places365 model...")
    places_model = torch.load(model_file, map_location=device)
    places_model.eval()

    categories_file = 'categories_places365.txt'
    if not os.path.exists(categories_file):
        log_error("Places365 categories file not found, downloading...")
        torch.hub.download_url_to_file(
            'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt',
            categories_file
        )
    with open(categories_file) as f:
        categories = [line.strip().split(' ')[0][3:] for line in f.readlines()]

    centre_crop = trn.Compose([
        trn.Resize((256, 256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    print("Places365 loaded successfully.")

def load_blip():
    global processor, blip_model
    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    blip_model.eval()
    print("BLIP model loaded successfully.")

def load_detr():
    global detr_feature_extractor, detr_model
    print("Loading DETR object detection model...")
    detr_feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
    detr_model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50').to(device)
    detr_model.eval()
    print("DETR model loaded successfully.")

def load_all_models():
    # Load each model with try/except
    try:
        load_blip()
    except Exception as e:
        log_error("Error loading BLIP model:")
        traceback.print_exc()
        global processor, blip_model
        processor, blip_model = None, None

    try:
        load_places365()
    except Exception as e:
        log_error("Error loading Places365 model or categories:")
        traceback.print_exc()
        global places_model, categories, centre_crop
        places_model, categories, centre_crop = None, [], None

    try:
        load_detr()
    except Exception as e:
        log_error("Error loading DETR model:")
        traceback.print_exc()
        global detr_feature_extractor, detr_model
        detr_feature_extractor, detr_model = None, None

def predict_scene(image_path):
    if places_model is None or centre_crop is None or not categories:
        return [("Unknown", 1.0)]
    print("Starting scene prediction...")
    img = Image.open(image_path)
    input_img = centre_crop(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = places_model.forward(input_img)
    probs = F.softmax(logits,1).squeeze()
    top5_idx = probs.topk(5).indices
    results = [(categories[idx], probs[idx].item()) for idx in top5_idx]
    print("Scene prediction done.")
    return results


def generate_caption(image_path):
    if processor is None or blip_model is None:
        return "No description available."
    print("Starting caption generation...")
    image = Image.open(image_path).convert('RGB')
    image = image.resize((320,320), Image.LANCZOS)
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = blip_model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    speak(f"The scene description is: {caption}")
    print("Caption generation done.")
    return caption

# Initialize TTS engine
tts_engine = pyttsx3.init()

# Configure TTS settings
tts_engine.setProperty('rate', 120)  # Adjust speech rate (default ~200, lower is slower)
tts_engine.setProperty('volume', 0.9)  # Adjust volume (0.0 to 1.0)


def speak(text):
    """Helper function to speak text using TTS."""
    tts_engine.say(text)
    tts_engine.runAndWait()

# Replace update_info with TTS enabled updates
def update_info(self, new_info):
    """Updates the info section and speaks the info."""
    self.info_label.text += f"{new_info}\n"
    speak(new_info.strip('[b][/b]'))  # Speak the info without markup tags




def detect_objects(image_path, score_threshold=0.9):
    if detr_model is None or detr_feature_extractor is None:
        print("DETR not available, skipping object detection.")
        return []
    print("Starting DETR object detection...")
    image = Image.open(image_path).convert("RGB")
    inputs = detr_feature_extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = detr_model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = detr_feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

    objects_detected = []
    img_width, img_height = image.size
    img_area = img_width * img_height

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score.item() > score_threshold:
            cls_name = detr_model.config.id2label[label.item()]
            # Simple distance estimation:
            x_min, y_min, x_max, y_max = box.tolist()
            box_area = (x_max - x_min) * (y_max - y_min)
            area_ratio = box_area / img_area

            if area_ratio > 0.3:
                distance = "close"
            elif area_ratio < 0.05:
                distance = "far"
            else:
                distance = "medium distance"

            objects_detected.append(f"{cls_name} ({distance})")
    speak(f"Objects detected are: {objects_detected}")  # Add TTS output
    print("Object detection done.")
    return objects_detected



class VisualAssistanceApp(App):
    def build(self):
        self.title = "Visual Assistance Application"
        self.continuous_mode = False
        # Load models at startup
        load_all_models()
        print("Opening ..")

        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        title_label = Label(
                            text="[b][i]Visual Assistance for Blind People[/i][/b]",
                            font_size='30sp',
                            color=(1,1,1,1),
                            size_hint=(1,0.1),
                            halign="center",
                            valign="middle",
                            markup=True 
                        )
        
        title_label.bind(size=self._update_text_size)
        main_layout.add_widget(title_label)
        middle_layout = BoxLayout(orientation='horizontal', spacing=10)

        camera_layout = BoxLayout(size_hint=(0.5,1))
        self.camera_widget = KivyImage()
        camera_layout.add_widget(self.camera_widget)

        # Info section with ScrollView
        info_layout = BoxLayout(orientation='vertical', size_hint=(0.5,1))
        self.info_scroll = ScrollView(size_hint=(1,1))
        self.info_label = Label(text="", font_size='18sp', color=(1,1,1,1), markup=True, 
                                size_hint_y=None, halign="left", valign="top")
        self.info_label.bind(texture_size=self._update_label_height)
        self.info_label.text_size = (self.info_scroll.width-20, None)
        self.info_scroll.add_widget(self.info_label)
        info_layout.add_widget(self.info_scroll)

        middle_layout.add_widget(camera_layout)
        middle_layout.add_widget(info_layout)
        main_layout.add_widget(middle_layout)

        # Bottom bar for buttons
        bottom_layout = BoxLayout(orientation='horizontal', size_hint=(1,0.12), spacing=10)

        self.capture_button = Button(text="Capture", font_size='20sp', background_color=(0,0.5,0.5,1))
        self.capture_button.bind(on_press=self.capture_and_analyze)
        bottom_layout.add_widget(self.capture_button)

        self.continuous_button = Button(text="Continuous (Off)", font_size='20sp', background_color=(0.5,0,0.5,1))
        self.continuous_button.bind(on_press=self.toggle_continuous_mode)
        bottom_layout.add_widget(self.continuous_button)

        help_button = Button(text="Help", font_size='20sp', background_color=(0,0,0.5,1))
        help_button.bind(on_press=self.help_user)
        bottom_layout.add_widget(help_button)

        exit_button = Button(text="Exit", font_size='20sp', background_color=(0.5,0,0,1))
        exit_button.bind(on_press=self.exit_app)
        bottom_layout.add_widget(exit_button)

        main_layout.add_widget(bottom_layout)

        self.analysis_lock = threading.Lock()

        # Initialize camera
        try:
            print("Initializing camera...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.update_info("[b]Failed to open camera.[/b]")
                log_error("Camera failed to open.")
            else:
                self.update_info("[b]Camera Ready.[/b]")
                print("Camera opened successfully.")
                Clock.schedule_interval(self.update_frame, 1.0/15.0)
        except Exception as e:
            self.update_info(f"[b]Camera error: {e}[/b]")
            log_error("Camera initialization error:")
            traceback.print_exc()
            self.cap = None

        return main_layout

    def _update_label_height(self, *args):
        self.info_label.text_size = (self.info_scroll.width - 20, None)
        self.info_label.texture_update()
        self.info_label.height = self.info_label.texture_size[1]

    def _update_text_size(self, instance, value):
        instance.text_size = (instance.width - 20, None)

    def update_frame(self, dt):
        if self.cap:
            ret, frame = self.cap.read()
            if not ret:
                log_error("Failed to read from camera.")
                return
            self.display_frame(frame)

            if self.continuous_mode:
                current_time = time.time()
                if not hasattr(self, 'last_analysis_time'):
                    self.last_analysis_time = current_time
                if current_time - self.last_analysis_time > 5: # every 5 seconds
                    if self.analysis_lock.acquire(blocking=False):
                        self.last_analysis_time = current_time
                        self.analyze_frame(frame, continuous=True)

    def display_frame(self, frame):
        try:
            buf = cv2.flip(frame, 0).tobytes()
            h, w, ch = frame.shape
            texture = Texture.create(size=(w,h), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.camera_widget.texture = texture
        except Exception as e:
            log_error("Error displaying frame:")
            traceback.print_exc()

    def capture_and_analyze(self, instance):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                if self.analysis_lock.acquire(blocking=False):
                    self.analyze_frame(frame, continuous=False)
                else:
                    self.update_info("[b]Analysis in progress...[/b]")
                    speak("Analysis is in progress. Please wait.")  # Speech during ongoing analysis
            else:
                self.update_info("[b]Failed to capture frame.[/b]")
                log_error("Failed to capture frame on button click.")
                speak("Failed to capture the frame.")  # Speak error message
        else:
            self.update_info("[b]Camera not initialized.[/b]")
            log_error("Capture requested but camera not initialized.")
            speak("The camera is not initialized.")  # Speak error message

    def toggle_continuous_mode(self, instance):
        self.continuous_mode = not self.continuous_mode
        instance.text = f"Continuous ({'On' if self.continuous_mode else 'Off'})"
        mode_status = "Continuous mode on" if self.continuous_mode else "Continuous mode off"
        self.update_info(f"[b]{mode_status}[/b]")
        print(mode_status)

    def help_user(self, instance):
        help_text = ("[b]Help / Instructions:[/b]\n"
                     "Capture: Takes a snapshot and analyzes the scene.\n"
                     "Continuous: Automatically analyze every few seconds.\n"
                     "Help: Show these instructions.\n"
                     "Exit: Close the application.\n\n"
                     "Features:\n"
                     "- Scene recognition (Places365)\n"
                     "- Image captioning (BLIP)\n"
                     "- OCR (Tesseract-based)\n"
                     "- Object detection & distance approximation (DETR)\n\n"
                     "Note: OCR and heavy models may not run smoothly on all Android devices.\n")
        self.update_info(help_text)

    def analyze_frame(self, frame, continuous=False):
        # Run analysis in background
        thread = threading.Thread(target=self.analyze_frame_in_background, args=(frame, continuous), daemon=True)
        thread.start()

    def analyze_frame_in_background(self, frame, continuous):
        try:
            print("Starting analysis...")
            img_path = "captured_frame.jpg"
            small_frame = cv2.resize(frame, (640,480), interpolation=cv2.INTER_AREA)
            cv2.imwrite(img_path, small_frame)

            scene_description = ""
            caption = "No description available."
            text_detected = ""
            objects_detected = []

            # Scene Recognition
            if places_model and centre_crop and categories:
                try:
                    predictions = predict_scene(img_path)
                    scene_description = "\n".join([f"{scene}: {prob:.2f}" for scene, prob in predictions])
                except Exception as e:
                    log_error("Scene prediction failed unexpectedly:")
                    traceback.print_exc()
                    scene_description = "Unknown scene"

            # Caption
            if processor and blip_model:
                try:
                    caption = generate_caption(img_path)
                except Exception as e:
                    log_error("Caption generation failed unexpectedly:")
                    traceback.print_exc()

            # OCR
            try:
                print("Starting OCR...")
                text_detected = pytesseract.image_to_string(img_path)
                print("OCR done.")
            except Exception as e:
                log_error("Error during OCR:")
                traceback.print_exc()
                text_detected = ""

            # DETR Object Detection
            if detr_model and detr_feature_extractor:
                try:
                    objects_detected = detect_objects(img_path, score_threshold=0.9)
                except Exception as e:
                    log_error("Object detection failed unexpectedly:")
                    traceback.print_exc()

            info = (f"[b]Scene Recognition:[/b]\n{scene_description if scene_description else 'Unknown'}\n\n"
                    f"[b]Caption:[/b] {caption}\n\n"
                    f"[b]Text Detected:[/b] {text_detected.strip() if text_detected.strip() else 'None'}\n\n"
                    f"[b]Objects Detected:[/b]\n{', '.join(objects_detected) if objects_detected else 'None'}")

            print("Analysis completed successfully.")
            Clock.schedule_once(lambda dt: self.update_info(info), 0)
        except Exception as e:
            log_error("Unexpected error in analyze_frame_in_background:")
            traceback.print_exc()
            fallback_info = "[b]An unexpected error occurred during analysis. Please try again.[/b]"
            Clock.schedule_once(lambda dt: self.update_info(fallback_info), 0)
        finally:
            if self.analysis_lock.locked():
                self.analysis_lock.release()

    def update_info(self, text):
        self.info_label.text = text

    def exit_app(self, instance):
        print("Exiting application...")
        App.get_running_app().stop()
        sys.exit(0)  # Ensure full termination

    def on_stop(self):
        if self.cap:
            self.cap.release()



if __name__ == "__main__":
    try:
        VisualAssistanceApp().run()
    except Exception as e:
        log_error("Uncaught exception in main:")
        traceback.print_exc()

        