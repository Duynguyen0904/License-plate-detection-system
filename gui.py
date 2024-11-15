import os
import sys
import cv2
import easyocr
import numpy as np
from PyQt5.QtCore import Qt
from ultralytics import YOLO
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QWidget, QFileDialog, QGroupBox, QScrollArea


EASY_OCR = easyocr.Reader(['en']) ## initiating easyocr
OCR_TH = 0.2

class LicensePlateGui(QWidget):
    def __init__(self):
        super().__init__()

        # Import model
        self.model = YOLO('last.pt')
        self.classes = self.model.names

        # Setup main window
        self.setWindowTitle('License Plate Detector')
        self.setGeometry(100, 100, 800, 600) ## adjust width to fit layout 
        self.setWindowIcon(QIcon("assets//gui.ico"))

        # Source image display area
        self.source_image_label = QLabel("Choose image")
        self.source_image_label.setFixedSize(700, 700)
        self.source_image_label.setStyleSheet("border: 1px solid black")
        self.source_image_label.setAlignment(Qt.AlignCenter)


        # Result group box to enclose result fields
        result_group_box = QGroupBox("Result")
        result_layout = QVBoxLayout()

        # Plate result
        plate_layout = QHBoxLayout()

        self.plate_label = QLabel("Plate")
        self.plate_label.setFixedWidth(70)
        plate_layout.addWidget(self.plate_label)

        self.plate_result = QLabel("")
        self.plate_result.setStyleSheet("border: 1px solid black; padding: 5px;")
        self.plate_result.setFixedSize(150, 80)
        self.plate_result.setAlignment(Qt.AlignCenter)
        plate_layout.addWidget(self.plate_result)

        # Character Result
        character_layout = QHBoxLayout()

        self.character_label = QLabel("Character")
        self.character_label.setFixedWidth(70)
        character_layout.addWidget(self.character_label)

        self.character_result = QLabel("")
        self.character_result.setStyleSheet("border: 1px solid black; padding: 5px;")
        self.character_result.setFixedSize(150, 80)
        self.character_result.setAlignment(Qt.AlignCenter)
        character_layout.addWidget(self.character_result)

        # Add All Result Fields to the Result Layout
        result_layout.addLayout(plate_layout)
        result_layout.addLayout(character_layout)

        # Set the layout for the result group box
        result_group_box.setLayout(result_layout)


        # Control Bar with Buttons Enclosed in Group Box
        control_group_box = QGroupBox("Control Bar")
        # control_group_box.setFixedHeight(200)
        # control_group_box.setFixedWidth(300)
        control_layout = QVBoxLayout()

        # Create function buttons
        load_button = QPushButton("Load Image")
        load_button.clicked.connect(self.load_image)
        control_layout.addWidget(load_button)

        process_button = QPushButton("Processing")
        process_button.clicked.connect(self.process_image)
        control_layout.addWidget(process_button)

        clear_button = QPushButton("Clear Box")
        clear_button.clicked.connect(self.clear_points)
        control_layout.addWidget(clear_button)
        
        # control_layout.addStretch()  # Add stretch to push buttons to the top

        # Set the layout for the control group box
        control_group_box.setLayout(control_layout)


        # Response Group Box to enclose Result Fields
        response_group_box = QGroupBox("Response")
        # response_group_box.setFixedHeight(200)
        # response_group_box.setFixedWidth(300)
        response_layout = QVBoxLayout()

        # Create the label and set style
        self.label = QLabel(self)
        self.label.setStyleSheet("border: 1px solid black; padding: 5px;")
        self.label.setAlignment(Qt.AlignTop)
        self.label.setWordWrap(True)

        # Create a scroll area and add the label to it
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)  # Allow the scroll area to adjust as text grows
        self.scroll_area.setWidget(self.label)  # Set the label as the scrollable widget
        response_layout.addWidget(self.scroll_area)

        # Set the layout for the result group box
        response_group_box.setLayout(response_layout)


        # Main Layout with Spanning and Positioning
        main_layout = QGridLayout()
        main_layout.addWidget(self.source_image_label, 0, 0, 3, 2)     # Src Image spans rows 0-2 and columns 0-1
        main_layout.addWidget(result_group_box, 0, 2)               # Result box in column 2, row 0
        main_layout.addWidget(response_group_box, 2, 2)             # Response box in column 2, row 1
        main_layout.addWidget(control_group_box, 1, 2)              # Control box in column 2, row 2

        # Set the layout to the main window
        self.setLayout(main_layout)

    
    def load_image(self):
        # Open a file dialog to load an image
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)")

        if file_name:
            self.image_path = file_name
            self.original_image = cv2.imread(self.image_path)
            image_name = os.path.basename(self.image_path)
            image_name = f"[!] {image_name} was loaded."
            self.update_message(image_name)
            self.update_image_display(self.original_image, self.source_image_label)

    def update_image_display(self, image, place):
        """Display the given image on QLabel"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = channel * width
        q_img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pix_map = QPixmap.fromImage(q_img)
        place.setPixmap(pix_map.scaled(place.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def detect_obj(self, frame):
        results = self.model.predict(frame)
        labels, coordinates = results[0].boxes.data[:, -1], results[0].boxes.data[:, :-1]
        return labels, coordinates

    def plot_boxes(self, frame, results):
        labels, coord = results
        n = len(labels)
        ### looping through the detections
        for num_i in range(n): ## loop of available license plate appear
            row = coord[num_i]
            if row[4] >= 0.75: ### threshold value for detection. We are discarding everything below this value
                x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3]) ### BBox coordinates
                text_d = self.classes[int(labels[num_i])]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## for BBox
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
                cv2.putText(frame, f"{text_d}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
                
        return frame

    def recognize_plate_easyocr(self, nplate, reader, region_threshold):
        ### resize the crop image
        cropx_shape, cropy_shape = nplate.shape[1], nplate.shape[0]
        if (cropy_shape*1.5) < cropx_shape: #for rectangle plate
            nplate = cv2.resize(nplate,(400,200))
        # else:
        elif (cropy_shape*2) > cropx_shape: #for square plate
            nplate = cv2.resize(nplate,(300,250))
        
        # ## Preprocessing
        gray = cv2.cvtColor(nplate, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 3)

        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        nplate = 255 - opening
        nplate = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 47, 14)
        nplate = gray
        
        ## ocr reading 
        ocr_result = reader.readtext(nplate, paragraph = True)
        text = self.filter_text(region = nplate, ocr_result = ocr_result, region_threshold = region_threshold)

        if len(text) == 1:
            text = text[0].upper()
        return text   

    def filter_text(self, region, ocr_result, region_threshold):
        rectangle_size = region.shape[0]*region.shape[1]
        plate = []

        for result in ocr_result:
            length = np.sum(np.subtract(result[0][1], result[0][0]))
            height = np.sum(np.subtract(result[0][2], result[0][1]))
            if length*height / rectangle_size > region_threshold:
                plate.append(result[1])
        for detection in ocr_result:
            number_plate = f"[!] Number plate: {detection[1]}"
            self.update_message(number_plate)
        return plate

    def crop_license_plate(self, frame, results):
        labels, coord = results
        for i in range(len(labels)):
            row = coord[i]
            if row[4] >= 0.75:  # Detection threshold
                x1, y1, x2, y2 = map(int, row[:4])  # Bounding box coordinates
                return frame[y1:y2, x1:x2]  # Return cropped license plate image
        return None

    def process_image(self):
        # Check if an image is loaded
        if not hasattr(self, 'image_path'):
            self.update_message("[!] No image loaded")
            return
        
        # Read the image
        image = cv2.imread(self.image_path)
        results = self.detect_obj(image)

        self.source_image_label.clear()

        self.plot_image = self.plot_boxes(image, results)
        self.update_image_display(self.plot_image, self.source_image_label)
        
        # Crop the detected license plate
        self.license_plate_img = self.crop_license_plate(image, results)

        if self.license_plate_img is not None:
            # Convert the cropped license plate to QPixmap for display in the plate_result label
            plate_num  = self.recognize_plate_easyocr(nplate=self.license_plate_img, reader=EASY_OCR, region_threshold= OCR_TH)
            self.update_image_display(self.license_plate_img, self.plate_result)
            self.update_message("[!] License plate detected and displayed.")
            self.character_result.setText(str(plate_num))
        else:
            # Clear the plate result label if no license plate is detected
            self.plate_result.clear()
            self.plate_result.setText("No plate detected")
            self.character_result.clear()
            self.update_message("[!] No license plate detected.")

    def clear_points(self):
        self.label.clear()
        # self.update_message("[!] Message box cleared.")

    def update_message(self, image):
        # Append the new message to the existing text
        previous_text = self.label.text()
        self.label.setText(previous_text + image + "\n")

        # Scroll to the bottom to see the latest message
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LicensePlateGui()
    window.show()
    sys.exit(app.exec_())