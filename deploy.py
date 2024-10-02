### import the required libraries
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import os
from datetime import datetime

### define global variable
EASY_OCR = easyocr.Reader(['en']) ### initiating easyocr
OCR_TH = 0.2
i = int(input("Enter optional value: ")) ## optional variable

# Variable to store selected ROI points
points = []

###------------------------------function to run detection------------------------------------
def detect_obj(frame, model):
    frame = [frame]
    print(f"[!] Detecting...")
    results = model.predict(frame)
    labels, coordinates = results[0].boxes.data[:, -1], results[0].boxes.data[:, :-1]
    return labels, coordinates

###------------------------------function to plot BBox and results---------------------------
def plot_boxes(i, results, frame, classes):
    labels, coord = results
    n = len(labels)
    print(f"[!] Total {n} detections...")
    print(f"[!] Looping through all detections...")

    for num_i in range(n):
        row = coord[num_i]
        if row[4] >= 0.15:  # threshold value for detection
            print(f"[!] Extracting BBox coordinates...")
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])  # BBox coordinates
            coords = [x1, y1, x2, y2]
            text_d = classes[int(labels[num_i])]
            print(text_d)
            
            nplate = frame[int(y1):int(y2), int(x1):int(x2)]
            crop_write(num_i, i, crop=nplate)
            # crop_vid(crop=nplate)
            cv2.imshow(f"crop {num_i}", nplate)

            ### call the function to recognize ocr
            plate_num = recognize_plate_easyocr(nplate=nplate, coords=coords, reader=EASY_OCR, region_threshold=OCR_TH)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # BBox
            cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255, 0), -1)  # text label background
            cv2.putText(frame, f"{plate_num}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
    return frame

###---------------------------------function to recognize license plate---------------------------------------------------------------
def recognize_plate_easyocr(nplate, coords, reader, region_threshold):
    ### resize the cropped image
    cropx_shape, cropy_shape = nplate.shape[1], nplate.shape[0]
    if (cropy_shape * 1.5) < cropx_shape:  # for rectangular plate
        nplate = cv2.resize(nplate, (400, 200))
    elif (cropy_shape * 2) > cropx_shape:  # for square plate
        nplate = cv2.resize(nplate, (300, 250))
    
    # Preprocessing
    gray = cv2.cvtColor(nplate, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 3)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    nplate = 255 - opening
    nplate = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 47, 14)
    nplate = gray
    
    ## OCR reading
    ocr_result = reader.readtext(nplate, paragraph=True)
    text = filter_text(region=nplate, ocr_result=ocr_result, region_threshold=region_threshold)

    if len(text) == 1:
        text = text[0].upper()
    return text

###---------------------------------function to filter out wrong detections -------------------------------------------------------------------------------
def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0] * region.shape[1]
    plate = []
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        if length * height / rectangle_size > region_threshold:
            plate.append(result[1])
    for detection in ocr_result:
        print(f"[!] Number plate: {detection[1]}")
    return plate

###------------------------------------function to save cropped and detected images----------------------------------------------------
def img_write(i, frame):
    path = "detected_images/save_image"
    filename = 'saved_%s.jpg' % i
    fullpath = os.path.join(path, filename)
    cv2.imwrite(fullpath, frame)

def crop_write(n, i, crop):
    path = "detected_images/crop_image"
    filename = 'cropped_%s_%s.jpg' % (i, n)
    fullpath = os.path.join(path, filename)
    cv2.imwrite(fullpath, crop)

def crop_vid(crop):
    now = datetime.now()
    current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    path = "detected_images/crop_vid"
    filename = 'cropped_%s.jpg' % current_time
    fullpath = os.path.join(path, filename)
    cv2.imwrite(fullpath, crop)

###---------------------------------function to detect objects within the selected ROI-------------------------------------------------------------------------------
def select_points(event, x, y, flags, param):
    global points
    frame = param['frame']  # Access the frame from the param dictionary
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))  # Store the point
        if len(points) > 1:
            # Draw a line between the last two points
            cv2.line(frame, points[-2], points[-1], (0, 255, 0), 2)
        if len(points) == 4:
            # Close the polygon by connecting the last point to the first one
            cv2.line(frame, points[-1], points[0], (0, 255, 0), 2)
            cv2.imshow("ROI Selection", frame)
            cv2.waitKey(1)  # Add a small delay to allow the image to refresh
            cv2.destroyWindow("ROI Selection")  # Finish the selection

def run_detection_in_roi(frame, model, classes):
    if len(points) != 4:
        print("[ERROR] Please select exactly 4 points.")
        return frame
    
    # Draw the ROI polyLine on the final frame
    for i in range(len(points)):
        cv2.line(frame, points[i], points[(i + 1) % 4], (0, 255, 0), 2)

    # Create a mask for the polygonal ROI
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    roi_corners = np.array([points], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, 255)

    # Apply the mask to the frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Detect objects within the masked region
    results = detect_obj(masked_frame, model=model)
    frame = plot_boxes(i, results, frame, classes=classes)

    return frame

###-------------------------------------main function-------------------------------------------
def main(img_path=None, vid_path=None):
    print(f"[!] Loading model...")
    model = YOLO('license_plate.pt')
    classes = model.names

    ##----------------------------------------------for detection on image------------------------------------------
    if img_path is not None:
        print(f"[!] Working with image: {img_path}")
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detect_obj(frame, model=model)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = plot_boxes(i, results, frame, classes=classes)
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow('result', frame)
        img_write(i, frame=frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ###-----------------------------------------for detection on video------------------------
    elif vid_path is not None:
        print(f"[!] Working with video: {vid_path}")
        cap = cv2.VideoCapture(vid_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"[INFO] fps: {fps}")

        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read video frame.")
            return

        # Resize frame and allow ROI selection
        frame = cv2.resize(frame, (1000, 700))

        # Clone the frame for drawing polyLine without affecting the original frame
        clone_frame = frame.copy()
    
        cv2.imshow("ROI Selection", clone_frame)
        param = {'frame': frame}
        cv2.setMouseCallback("ROI Selection", select_points, param)
        cv2.waitKey(0)  # Wait until the user selects the ROI

        # Process video frames with selected ROI
        frame_no = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1000, 700))
            frame = run_detection_in_roi(frame, model, classes)

            cv2.imshow("Result", frame)
            if cv2.waitKey(25) & 0xFF == ord('e'):
                break
            frame_no += 1

        cap.release()
        cv2.destroyAllWindows()

###--------------------calling the main function----------------------
# main(vid_path="data/video/vid_%s.mp4"%i) ## for video
main(img_path = "data/image/%s.jpg" % i) ## for image
