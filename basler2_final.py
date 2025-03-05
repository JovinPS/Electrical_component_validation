import cv2
import time
import re
import pyttsx3
import logging
import threading
from tkinter import Tk, Label, Button, Entry, StringVar, DoubleVar
from paddleocr import PaddleOCR
from ultralytics import YOLO
from PIL import Image, ImageTk
from pypylon import pylon
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set logging level to ERROR to suppress debug and warning messages
# logging.getLogger('ppocr').setLevel(logging.ERROR)

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=True, show_log=True)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load custom YOLOv8 model
model = YOLO(r"C:\Users\jovin\Downloads\best_n2.pt")
model.to('cuda')

# Initialize variables
extracted_values = []
stable_value = None
display_start_time = None
display_duration = 3  # Duration to display the stable value in seconds
value_status = ""  # To store "OK" or "Not Good"

# Default acceptable range
magnitude = 0.0
tolerance = 0.0
lower_bound = magnitude - (tolerance / 100) * magnitude
upper_bound = magnitude + (tolerance / 100) * magnitude

# Text-to-speech function
def speak_status(status):
    engine.say(status)
    engine.runAndWait()

def detect_display_screen(frame):
    # Perform object detection
    results = model.predict(frame, conf=0.57, device="cuda")

    # Assuming the first detected object is the multimeter display
    if results[0].obb.xyxy.shape[0] > 0:
        x1, y1, x2, y2 =  map(int,results[0].obb.xyxy[0])
        # Ensure the coordinates are within the frame dimensions
        height, width, _ = frame.shape
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        roi = (x1, y1, x2, y2)
    else:
        roi = (100, 100, 400, 200)  # Fallback to a default ROI

    return roi

def extract_text_from_roi(frame, roi):
    x1, y1, x2, y2 = roi
    cropped_image = frame[y1:y2, x1:x2]
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    # Perform OCR on the grayscale image
    ocr_result = ocr.ocr(gray_image, cls=True)
    
    # Concatenate the detected text into a single string
    extracted_text_lines = []
    if ocr_result and ocr_result[0]:
        for line in ocr_result:
            for word in line:
                text, confidence = word[1]
                extracted_text_lines.append(text)

    extracted_text = ' '.join(extracted_text_lines)
    return extracted_text

# Update range values from user input
def update_range():
    global magnitude, tolerance, lower_bound, upper_bound
    try:
        magnitude = float(magnitude_var.get())
        tolerance = float(tolerance_var.get())
        lower_bound = magnitude - (tolerance / 100) * magnitude
        upper_bound = magnitude + (tolerance / 100) * magnitude
    except ValueError:
        magnitude = 50.0
        tolerance = 10.0
        lower_bound = magnitude - (tolerance / 100) * magnitude
        upper_bound = magnitude + (tolerance / 100) * magnitude

# Update GUI with the extracted and stable values
def update_gui():
    global stable_value, display_start_time, value_status
    
    if not camera.IsGrabbing():
        root.after(10, update_gui)
        return

    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        frame = image.GetArray()
        
        # Detect the display screen
        roi = detect_display_screen(frame)

        # Draw the ROI rectangle on the frame
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
        
        # Extract text from the ROI
        extracted_text = extract_text_from_roi(frame, roi)
        
        # Extract numerical values from the extracted text using regex
        numerical_values = re.findall(r"[-+]?\d*\.\d+|\d+", extracted_text)
        #cv2.putText(frame, f'OCR Text: {extracted_text}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert the first found numerical value to float if possible
        if numerical_values:
            try:
                value = float(numerical_values[0])
            except ValueError:
                value = 0.0  # Default to 0.0 if conversion fails
        else:
            value = 0.0  # Default to 0.0 if no numerical value is found

        extracted_values.append(value)

        # Check for stability
        if len(extracted_values) >= 10:  # Require at least 10 values for stability check
            last_values = extracted_values[-10:]
            if all(v == last_values[0] and v != 0.0 for v in last_values):
                stable_value = last_values[0]
                display_start_time = time.time()  # Record the start time of displaying the stable value
                extracted_values.clear()  # Clear the array after extracting the stable value

                # Check the stable value against the acceptable range
                if lower_bound <= stable_value <= upper_bound:
                    value_status = "OK"
                else:
                    value_status = "Not Good"

                # Start a new thread for text-to-speech
                tts_thread = threading.Thread(target=speak_status, args=(value_status,))
                tts_thread.start()
        
        # Display the extracted stable value and status if available and within the display duration
        if stable_value is not None and (time.time() - display_start_time) < display_duration:
            stable_value_var.set(stable_value)
            status_var.set(value_status)
        elif stable_value is not None and (time.time() - display_start_time) >= display_duration:
            stable_value = None  # Reset after displaying for the specified duration
            stable_value_var.set("")
            status_var.set("")

        frame_resized = cv2.resize(frame, (640,520))
        # Convert the frame to ImageTk format
        img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        # img=cv2.resize(frame,1000,1000)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update the video frame in the GUI
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    
    grabResult.Release()
    # Call the update_gui function again after 10ms
    root.after(10, update_gui)

# Initialize Tkinter window
root = Tk()
root.title("Multimeter Display OCR")

# Set the window size to fit the screen
root.geometry("1024x780")
root.resizable(True, True)

# # Configure grid layout
root.grid_columnconfigure(2, minsize=100)
root.grid_rowconfigure(0, minsize=100)
root.grid_rowconfigure(3, weight=1)
root.grid_rowconfigure(4, weight=1)


# Create and place labels and entries for the magnitude and tolerance
magnitude_var = DoubleVar()
tolerance_var = DoubleVar()

Label(root, text="Magnitude:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
Entry(root, textvariable=magnitude_var).grid(row=0, column=1, padx=5, pady=5, sticky="w")

Label(root, text="Tolerance (%):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
Entry(root, textvariable=tolerance_var).grid(row=1, column=1, padx=5, pady=5, sticky="w")

Button(root, text="Update Range", command=update_range).grid(row=2, column=0, columnspan=2, padx=5, pady=5)

# Create and place labels for the stable value and status
stable_value_var = StringVar()
status_var = StringVar()

Label(root, text="Stable Value:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
Label(root, textvariable=stable_value_var, font=("Helvetica", 16)).grid(row=3, column=1, padx=5, pady=5, sticky="w")

Label(root, text="Status:").grid(row=4, column=0, padx=5, pady=5, sticky="e")
Label(root, textvariable=status_var, font=("Helvetica", 16)).grid(row=4, column=1, padx=5, pady=5, sticky="w")

# Create a label to display the video feed
video_label = Label(root)
video_label.grid(row=0, column=2, rowspan=5, padx=10, pady=10, sticky="nsew")


# Initialize the Basler camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Start the GUI update loop
root.after(0, update_gui)
root.mainloop()

# Release the camera
camera.StopGrabbing()
camera.Close()
