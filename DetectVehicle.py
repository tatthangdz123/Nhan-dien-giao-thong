import pymongo
import easyocr
from datetime import datetime
from ultralytics import YOLO
import cv2
import numpy as np
import re
import pytesseract
import os  # Để tạo thư mục lưu ảnh
# Kết nối MongoDB

try:
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["traffic_db"]
    plates_collection = db["license_plates"]  # Collection để lưu tất cả thông tin
    client.server_info()  # Kiểm tra kết nối

except pymongo.errors.ServerSelectionTimeoutError:
    print("Không thể kết nối với MongoDB, kiểm tra lại server!")
    exit(1)
class LicensePlateRecognizer:
    def __init__(self, languages=['en', 'vi']):
        # Keep EasyOCR for backup
        self.ocr = easyocr.Reader(languages)
        
        # Configure Tesseract for license plates
        self.tesseract_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        # Your existing plate patterns
        self.plate_patterns = [
            r'^\d{2}[A-Z]\d{4,5}$',    # 59F12345
            r'^\d{2}[A-Z]-\d{4,5}$',    # 59F-12345
            r'^\d{2}[A-Z]\s\d{4,5}$',   # 59F 12345
            r'^\d{2}\s[A-Z]\s\d{4,5}$', # 59 F 12345
            r'^\d{2}-[A-Z]-\d{4,5}$'    # 59-F-12345
        ]
        
        # Add plate layout configurations
        self.plate_layouts = {
            "single_line": {"expected_chars": 8},  # e.g., 59F12345
            "double_line": {"top_chars": 3, "bottom_chars": 4}  # Top: 59F, Bottom: 1234
        }
    
    def detect_plate_layout(self, plate_img):
        """Determine if plate is single line or double line"""
        h, w = plate_img.shape[:2]
        aspect_ratio = w / h
        
        # Vietnamese plates typically have different aspect ratios
        # Single line: width > 2 * height
        # Double line: width < 2 * height
        if aspect_ratio > 2.0:
            return "single_line"
        else:
            return "double_line"
    
    def preprocess_for_ocr(self, img, layout):
        """Enhance image for better OCR based on layout"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Noise removal
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Different processing for single vs double line
        if layout == "double_line":
            # For double line, separate top and bottom rows
            h, w = processed.shape
            top_half = processed[0:h//2, :]
            bottom_half = processed[h//2:h, :]
            
            # Process each half separately
            top_half = cv2.dilate(top_half, kernel, iterations=1)
            bottom_half = cv2.dilate(bottom_half, kernel, iterations=1)
            
            # Recombine
            processed = np.vstack((top_half, bottom_half))
        else:
            # For single line, dilate characters
            processed = cv2.dilate(processed, kernel, iterations=1)
        
        # Invert back for OCR
        processed = cv2.bitwise_not(processed)
        
        return processed
    
    def segment_plate(self, plate_img):
        """Segment license plate into parts based on layout"""
        layout = self.detect_plate_layout(plate_img)
        
        if layout == "double_line":
            h, w = plate_img.shape[:2]
            top_half = plate_img[0:h//2, :]
            bottom_half = plate_img[h//2:h, :]
            
            return {
                "layout": layout,
                "parts": [top_half, bottom_half]
            }
        else:
            # For single line, we might want to segment into regions
            # e.g., first 2-3 characters (province code) and rest (numbers)
            h, w = plate_img.shape[:2]
            province_part = plate_img[:, 0:int(w*0.4)]  # First 40% for province code
            number_part = plate_img[:, int(w*0.4):]     # Rest for numbers
            
            return {
                "layout": layout,
                "parts": [province_part, number_part]
            }
    
    def recognize_plate(self, img):
        """Recognize license plate text using OCR"""
        processed_images = self.preprocess_plate_image(img)
        if not processed_images:
            return None
        
        all_results = []
        
        for proc_img in processed_images:
            # Determine plate layout
            layout = self.detect_plate_layout(proc_img)
            
            # Process the image for better OCR
            enhanced = self.preprocess_for_ocr(proc_img, layout)
            
            # Save debug image
            cv2.imwrite(f"debug_enhanced_plate_{datetime.now().strftime('%H%M%S')}.jpg", enhanced)
            
            # Try Tesseract OCR with the enhanced image
            try:
                # Use different PSM based on layout
                if layout == "double_line":
                    # Split plate into top and bottom halves
                    h = enhanced.shape[0]
                    top_half = enhanced[0:h//2, :]
                    bottom_half = enhanced[h//2:h, :]
                    
                    # Process each half with Tesseract
                    top_text = pytesseract.image_to_string(top_half, config=r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                    bottom_text = pytesseract.image_to_string(bottom_half, config=r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                    
                    # Clean and combine
                    top_text = re.sub(r'[^A-Z0-9]', '', top_text).strip()
                    bottom_text = re.sub(r'[^A-Z0-9]', '', bottom_text).strip()
                    plate_text = f"{top_text}-{bottom_text}"
                else:
                    # For single line plates
                    plate_text = pytesseract.image_to_string(enhanced, config=self.tesseract_config)
                    plate_text = re.sub(r'[^A-Z0-9]', '', plate_text).strip()
                
                # Save result if it has reasonable length
                if len(plate_text) >= 5:
                    all_results.append((plate_text, 0.9))  # Confidence is arbitrary here
            except Exception as e:
                print(f"Tesseract OCR error: {e}")
            
            # Try EasyOCR as backup
            try:
                ocr_results = self.ocr.readtext(proc_img)
                for (_, text, conf) in ocr_results:
                    cleaned_text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
                    if len(cleaned_text) >= 5:
                        all_results.append((cleaned_text, conf))
            except Exception as e:
                print(f"EasyOCR error: {e}")
        
        # Sort results by confidence
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        if not all_results:
            return None
        
        # Get best result
        best_text, _ = all_results[0]
        
        # Format and validate
        formatted_text = self.format_plate_number(best_text)
        if self.is_valid_plate(formatted_text):
            return formatted_text
        elif len(all_results) > 1:
            # Try second best
            second_best, _ = all_results[1]
            formatted_second = self.format_plate_number(second_best)
            if self.is_valid_plate(formatted_second):
                return formatted_second
        
        # Return best match even if not valid
        return formatted_text
class TrafficLightClassifier:
    def __init__(self):
        # Định nghĩa dải giá trị HSV cho từng màu đèn
        self.lower_red1 = np.array([0, 100, 70])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 100, 70])
        self.upper_red2 = np.array([180, 255, 255])
        
        self.lower_yellow = np.array([15, 80, 80])
        self.upper_yellow = np.array([35, 255, 255])
        
        self.lower_green = np.array([40, 30, 40])
        self.upper_green = np.array([85, 255, 255])
        
        # Ngưỡng pixel tối thiểu
        self.min_pixel_count = 20
        
        # Mapping từ số sang tên màu
        self.color_map = {0: "RED", 1: "YELLOW", 2: "GREEN"}
        
    def classify(self, frame):
        # Chuyển đổi sang không gian màu HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Tạo mặt nạ cho từng màu
        mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_green = cv2.inRange(hsv, self.lower_green, self.upper_green)
        
        # Đếm số pixel cho mỗi màu
        red_count = cv2.countNonZero(mask_red)
        yellow_count = cv2.countNonZero(mask_yellow)
        green_count = cv2.countNonZero(mask_green)
        
        # Debug: in ra số lượng pixel của mỗi màu
        print(f"Red: {red_count}, Yellow: {yellow_count}, Green: {green_count}")
        
        # Tạo danh sách đếm pixel
        counts = [red_count, yellow_count, green_count]
        
        if max(counts) < self.min_pixel_count:
            return None
        
        color_id = counts.index(max(counts))
        return color_id, self.color_map[color_id]

class LicensePlateRecognizer:
    def __init__(self, languages=['en', 'vi']):
        # Khởi tạo EasyOCR với ngôn ngữ Việt và tiếng Anh
        self.ocr = easyocr.Reader(languages)
        
        # Mẫu biểu thức chính quy cho biển số xe Việt Nam
        # Format: 2 chữ số - 1 chữ cái - 5 chữ số hoặc 2 chữ số - 1 chữ cái - 4 chữ số
        self.plate_patterns = [
            r'^\d{2}[A-Z]\d{4,5}$',    # 59F12345
            r'^\d{2}[A-Z]-\d{4,5}$',    # 59F-12345
            r'^\d{2}[A-Z]\s\d{4,5}$',   # 59F 12345
            r'^\d{2}\s[A-Z]\s\d{4,5}$', # 59 F 12345
            r'^\d{2}-[A-Z]-\d{4,5}$'    # 59-F-12345
        ]
    
    def preprocess_plate_image(self, img):
        # Kiểm tra xem img có tồn tại và không rỗng
        if img is None or img.size == 0:
            return None
            
        # Chuyển sang ảnh grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Áp dụng GaussianBlur để giảm nhiễu
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Ngưỡng hóa Adaptive Threshold
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Dilation để làm rõ các ký tự
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        # Resize ảnh lớn hơn để tăng độ chính xác OCR
        scale_factor = 2.0
        resized = cv2.resize(dilated, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # Tạo bounding box quanh biển số
        contours, _ = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
        
        # Tạo các phiên bản tiền xử lý khác nhau để tăng khả năng nhận diện
        processed_images = [
            resized,
            gray,
            cv2.equalizeHist(gray),  # Cân bằng histogram
            thresh,
            dilated
        ]
        
        return processed_images
    
    def is_valid_plate(self, text):
        # Loại bỏ ký tự đặc biệt và chuyển thành chữ viết hoa
        cleaned_text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
        
        # Kiểm tra xem có phải biển số hợp lệ
        for pattern in self.plate_patterns:
            if re.match(pattern, cleaned_text):
                return True
                
        # Kiểm tra nếu chuỗi chứa đủ số và chữ
        if re.match(r'.*\d+.*[A-Z]+.*\d+.*', cleaned_text) and len(cleaned_text) >= 7:
            return True
            
        return False
    
    def format_plate_number(self, text):
        # Loại bỏ khoảng trắng và ký tự đặc biệt
        cleaned_text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
        
        # Nếu độ dài từ 7-10 ký tự, có thể là biển số xe
        if 7 <= len(cleaned_text) <= 10:
            # Tạo định dạng chuẩn: 2 chữ số - 1 chữ cái - số còn lại
            if re.match(r'^\d{2}[A-Z]', cleaned_text[:3]):
                return f"{cleaned_text[:2]}{cleaned_text[2:3]}-{cleaned_text[3:]}"
        
        return cleaned_text
    
    def recognize_plate(self, img):
        processed_images = self.preprocess_plate_image(img)
        if not processed_images:
            return None
            
        all_results = []
        confidence_threshold = 0.4
        
        # Thử nhận dạng với mỗi phiên bản tiền xử lý
        for proc_img in processed_images:
            try:
                results = self.ocr.readtext(proc_img)
                for (bbox, text, conf) in results:
                    if conf > confidence_threshold:
                        # Lọc và chỉ giữ chữ và số
                        cleaned_text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
                        if len(cleaned_text) >= 5:  # Biển số thường có ít nhất 5 ký tự
                            all_results.append((cleaned_text, conf))
            except Exception as e:
                print(f"Lỗi OCR: {e}")
                continue
        
        # Sắp xếp kết quả theo độ tin cậy
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        if not all_results:
            return None
            
        # Lấy kết quả có độ tin cậy cao nhất
        best_text, best_conf = all_results[0]
        
        # Kiểm tra và định dạng biển số
        if self.is_valid_plate(best_text):
            return self.format_plate_number(best_text)
        elif len(all_results) > 1:
            # Thử kết quả có độ tin cậy cao thứ hai
            second_best, _ = all_results[1]
            if self.is_valid_plate(second_best):
                return self.format_plate_number(second_best)
        
        # Trả về kết quả tốt nhất dù không đạt chuẩn định dạng
        return self.format_plate_number(best_text)

class ObjectDetector:
    def __init__(self, object_model_path, plate_model_path):
        self.object_model = YOLO(object_model_path)
        self.plate_model = YOLO(plate_model_path)
        self.plate_recognizer = LicensePlateRecognizer(['en', 'vi'])
        self.traffic_light_classifier = TrafficLightClassifier()

        # Định nghĩa class theo model đã train
        self.vehicle_types = {0: "BUS", 1: "CAR", 2: "MOTORBIKE", 4: "TRUCK"}
        self.traffic_light_id = 3  # Đèn giao thông
        
        # Cập nhật thông tin từ file YAML - chỉ có 1 class là biển số xe
        self.plate_class_id = 0  # Biển số xe là class duy nhất (index 0)
        
        # Biến theo dõi trạng thái đèn hiện tại
        self.current_light_color = None
        
        # Hỗ trợ phát hiện đèn khi mô hình không nhận diện được
        self.manual_traffic_light_detection = True
        
        # Lưu ảnh debug
        self.save_debug_frames = True
        self.frame_count = 0
        
        # Theo dõi biển số đã nhận diện để tránh trùng lặp
        self.detected_plates = set()
        
        # Lưu lại kết quả nhận diện biển số
        self.plate_history = {}

    def detect_objects(self, frame):
        return self.object_model(frame)[0]
    
    def detect_plate(self, vehicle_crop, x1, y1, frame, vehicle_type):
        plate_results = self.plate_model(vehicle_crop)
        detected_plates = []
    
    # Kiểm tra xem có phát hiện biển số không
        if len(plate_results[0].boxes.xyxy) == 0:
        # Nếu không phát hiện biển bằng YOLO, thử nhận dạng trực tiếp từ vùng xe
            if vehicle_type in ["CAR", "TRUCK", "BUS"]:  # Chỉ áp dụng cho các loại xe lớn
            # Lấy 1/3 phía dưới của xe để tìm biển số
                h, w = vehicle_crop.shape[:2]
                plate_region = vehicle_crop[int(h*2/3):h, int(w/4):int(w*3/4)]
            
                if plate_region.size > 0:
                    plate_text = self.plate_recognizer.recognize_plate(plate_region)
                    if plate_text:
                        detected_plates.append(plate_text)
                    
                    # THÊM ĐOẠN MÃ NÀY: Lưu vào lịch sử biển số và MongoDB
                        self.plate_history[plate_text] = {
                            'last_seen': datetime.now(),
                            'vehicle_type': vehicle_type,
                            'confidence': 0.7  # Đặt độ tin cậy giả định
                        }
                        self.save_plate_to_db(plate_text, vehicle_type, plate_region)
                    
                        # Vẽ hộp chứa vùng biển số
                        y_offset = int(h*2/3)
                        x_offset = int(w/4)
                        cv2.rectangle(frame, 
                                (x1 + x_offset, y1 + y_offset), 
                                (x1 + int(w*3/4), y1 + h), 
                                (0, 255, 255), 2)
                        cv2.putText(frame, plate_text, (x1 + x_offset, y1 + y_offset - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                return detected_plates  
    def save_plate_to_db(self, plate_text, vehicle_type, plate_img=None):
        # Chuyển ảnh thành base64 để lưu vào MongoDB (tùy chọn)
        img_encoded = None
        if plate_img is not None and plate_img.size > 0:
            try:
                _, buffer = cv2.imencode('.jpg', plate_img)
                img_encoded = buffer.tobytes()
            except Exception as e:
                print(f"Lỗi khi mã hóa ảnh: {e}")
        
        # Kiểm tra xem biển số đã tồn tại chưa
        existing_plate = plates_collection.find_one({"plate_number": plate_text})
        
        if existing_plate:
            # Cập nhật thông tin biển số đã tồn tại
            plates_collection.update_one(
                {"plate_number": plate_text},
                {"$set": {
                    "last_seen": datetime.now(),
                    "vehicle_type": vehicle_type,
                    "occurrences": existing_plate.get("occurrences", 0) + 1
                }}
            )
        else:
            # Thêm biển số mới
            plate_data = {
                "plate_number": plate_text,
                "vehicle_type": vehicle_type,
                "first_seen": datetime.now(),
                "last_seen": datetime.now(),
                "occurrences": 1
            }
            
            # Chỉ lưu hình ảnh nếu có
            if img_encoded:
                plate_data["image"] = img_encoded
                
            plates_collection.insert_one(plate_data)

    def classify_traffic_light(self, frame, x1, y1, x2, y2):
        # Cắt vùng đèn giao thông
        light_crop = frame[y1:y2, x1:x2].copy()
        if light_crop.size == 0:
            return None
        
        # Sử dụng bộ phân loại để xác định màu đèn
        color_result = self.traffic_light_classifier.classify(light_crop)
        if color_result:
            color_id, color_name = color_result
            self.current_light_color = color_id
            return color_name
        return None
    
    def manual_detect_traffic_light(self, frame):
        # Phát hiện đèn giao thông theo vùng cố định khi model không nhận diện được
        h, w = frame.shape[:2]
        
        # Vùng ROI cố định
        x1, y1 = int(w * 0.85), int(h * 0.2)
        x2, y2 = int(w * 0.99), int(h * 0.4)
        
        # Đánh dấu vùng ROI để debug
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
        light_crop = frame[y1:y2, x1:x2].copy()
        
        # Hiển thị vùng crop để debug nếu cần
        if self.save_debug_frames and self.frame_count % 30 == 0:  # Lưu mỗi 30 frame
            cv2.imwrite(f"debug_light_region_{self.frame_count}.jpg", light_crop)
            
            # Hiển thị mask màu xanh để debug
            hsv = cv2.cvtColor(light_crop, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, self.traffic_light_classifier.lower_green, 
                                    self.traffic_light_classifier.upper_green)
            cv2.imwrite(f"debug_green_mask_{self.frame_count}.jpg", green_mask)
            
            # Mask màu đỏ và vàng để so sánh
            red_mask1 = cv2.inRange(hsv, self.traffic_light_classifier.lower_red1, 
                                   self.traffic_light_classifier.upper_red1)
            red_mask2 = cv2.inRange(hsv, self.traffic_light_classifier.lower_red2, 
                                   self.traffic_light_classifier.upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            yellow_mask = cv2.inRange(hsv, self.traffic_light_classifier.lower_yellow, 
                                     self.traffic_light_classifier.upper_yellow)
            
            cv2.imwrite(f"debug_red_mask_{self.frame_count}.jpg", red_mask)
            cv2.imwrite(f"debug_yellow_mask_{self.frame_count}.jpg", yellow_mask)
            
        self.frame_count += 1
        
        color_result = self.traffic_light_classifier.classify(light_crop)
        
        if color_result:
            color_id, color_name = color_result
            self.current_light_color = color_id
            
            # Vẽ rectangle với màu phù hợp
            color_code = (0, 0, 255) if color_name == "RED" else \
                         (0, 255, 255) if color_name == "YELLOW" else \
                         (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_code, 2)
            cv2.putText(frame, f"TRAFFIC_LIGHT: {color_name}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_code, 2)
            
            return color_name
        
        return None

    def save_violation(self, vehicle_type, license_plates, violation_img=None):
        timestamp = datetime.now()
    
        for plate in license_plates:
        # Convert the full frame image to binary format for MongoDB storage
            img_encoded = None
            if violation_img is not None and violation_img.size > 0:
                try:
                    _, buffer = cv2.imencode('.jpg', violation_img)
                    img_encoded = buffer.tobytes()
                except Exception as e:
                    print(f"Lỗi khi mã hóa ảnh vi phạm: {e}")
        
        # Update or insert plate information with violation status
            update_data = {
            "$set": {
                "last_seen": timestamp,
                "vehicle_type": vehicle_type,
                "last_violation": timestamp,
                "violation_type": "RED_LIGHT"
            },
            "$inc": {"occurrences": 1, "violations": 1}
            }
        
        # Only add image if successfully encoded
            if img_encoded:
                update_data["$set"]["violation_image"] = img_encoded
            
            plates_collection.update_one(
                {"plate_number": plate},
                update_data,
                upsert=True
        )

class RedLightViolationDetector:
    def __init__(self, width, height):
        self.detect_line_y = int(height * 0.29)
        self.stop_line_y = int(height * 0.49)
        self.detect_margin = int(width * 0.25)
        self.stop_margin = int(width * 0.12)
    
    def check_violation(self, y2, is_red_light):
        # Chỉ báo vi phạm nếu đèn đỏ VÀ xe vượt qua vạch
        return is_red_light and y2 <= self.stop_line_y
    
    def draw_lines(self, frame, width):
        cv2.line(frame, (self.detect_margin, self.detect_line_y), 
                        (width - self.detect_margin, self.detect_line_y), 
                        (255, 255, 0), 2)
        cv2.line(frame, (self.stop_margin, self.stop_line_y), 
                        (width - self.stop_margin, self.stop_line_y), 
                        (0, 0, 255), 3)

def process_frame(frame, detector, violation_detector, width):
    result = detector.detect_objects(frame)
    
    # Đầu tiên, tìm và phân loại đèn giao thông
    is_red_light = False
    light_color = None
    traffic_light_found = False
    
    for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), 
                              result.boxes.cls.cpu().numpy(), 
                              result.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)
        
        # Nếu phát hiện đèn giao thông, phân loại màu của nó
        if class_id == detector.traffic_light_id:
            traffic_light_found = True
            print("Đèn giao thông phát hiện được!")
            light_color = detector.classify_traffic_light(frame, x1, y1, x2, y2)
            if light_color:
                is_red_light = (light_color == "RED")
                color_code = (0, 0, 255) if light_color == "RED" else \
                            (0, 255, 255) if light_color == "YELLOW" else \
                            (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_code, 2)
                cv2.putText(frame, f"TRAFFIC_LIGHT: {light_color}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_code, 2)
    
    # Nếu không tìm thấy đèn bằng YOLO, thử phương pháp thủ công
    if not traffic_light_found and detector.manual_traffic_light_detection:
        light_color = detector.manual_detect_traffic_light(frame)
        if light_color:
            is_red_light = (light_color == "RED")
    
    # Hiển thị trạng thái đèn hiện tại trên màn hình với màu tương ứng
    if light_color == "RED":
        status_text = "RED LIGHT"
        status_color = (0, 0, 255)  # BGR cho màu đỏ
    elif light_color == "YELLOW":
        status_text = "YELLOW LIGHT"
        status_color = (0, 255, 255)  # BGR cho màu vàng
    elif light_color == "GREEN":
        status_text = "GREEN LIGHT"
        status_color = (0, 255, 0)  # BGR cho màu xanh
    else:
        status_text = "NO TRAFFIC LIGHT DETECTED"
        status_color = (255, 255, 255)  # BGR cho màu trắng
    
    cv2.putText(frame, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    # Sau đó xử lý các phương tiện
    for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), 
                              result.boxes.cls.cpu().numpy(), 
                              result.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)

        if class_id in detector.vehicle_types and y2 >= violation_detector.detect_line_y:
            detect_vehicle(frame, detector, violation_detector, x1, y1, x2, y2, class_id, is_red_light)

def detect_vehicle(frame, detector, violation_detector, x1, y1, x2, y2, class_id, is_red_light):
    vehicle_name = detector.vehicle_types[class_id]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, vehicle_name, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    vehicle_crop = frame[y1:y2, x1:x2]
    plates_detected = detector.detect_plate(vehicle_crop, x1, y1, frame, vehicle_name)
    
    if violation_detector.check_violation(y2, is_red_light):
        cv2.putText(frame, "VUOT DEN DO!", (x1, y2 + 20),

                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    if plates_detected:
        # Capture the full frame when a red light violation is detected
        violation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_frame_copy = frame.copy()
        
        # Save violation image with timestamp to file system
        violation_img_path = f"violations/violation_{violation_timestamp}.jpg"
        os.makedirs("violations", exist_ok=True)
        cv2.imwrite(violation_img_path, full_frame_copy)
        
        # Save violation information to MongoDB with the full frame image
        detector.save_violation(vehicle_name, plates_detected, full_frame_copy)
        (cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        if plates_detected:
            # Capture the full frame when a red light violation is detected
            full_frame_copy = frame.copy()
            
            # Save violation information to MongoDB with the full frame image
            detector.save_violation(vehicle_name, plates_detected, full_frame_copy)


# Khởi tạo hệ thống
detector = ObjectDetector("Vehicle.pt", "LP.pt")
detector.plate_recognizer = LicensePlateRecognizer()
cap = cv2.VideoCapture("D:/DetectTraffic/Video/5P.mp4")
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
violation_detector = RedLightViolationDetector(width, height)

cv2.namedWindow("Object & Plate Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object & Plate Detection", width, height)

# Thêm các biến theo dõi
frame_count = 0
detected_green_frames = 0
last_10_frames = []

# Thêm hiển thị thống kê biển số
plate_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  
    
    frame_count += 1
    
    # Thêm một bản sao của frame gốc để so sánh nếu cần
    original_frame = frame.copy()
    
    process_frame(frame, detector, violation_detector, width)
    violation_detector.draw_lines(frame, width)
    
    # Theo dõi việc phát hiện đèn xanh
    if detector.current_light_color == 2:  # GREEN
        detected_green_frames += 1
        last_10_frames.append(1)
    else:
        last_10_frames.append(0)
        
    # Chỉ giữ 10 frame gần nhất
    if len(last_10_frames) > 10:
        last_10_frames.pop(0)
        
    # Cập nhật số lượng biển số đã phát hiện
    plate_count = len(detector.plate_history)
    
    # Hiển thị thống kê
    cv2.putText(frame, f"Plates Detected: {plate_count}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    cv2.imshow("Object & Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
client.close()
cv2.destroyAllWindows()