import cv2
import numpy as np
from pymongo import MongoClient

# Kết nối MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["traffic_db"]
plates_collection = db["license_plates"]

# Truy xuất dữ liệu ảnh từ MongoDB (Thay biển số bằng biển số thực tế)
plate_record = plates_collection.find_one({"plate_number": "15326"})

if plate_record and "violation_image" in plate_record:
    # Lấy dữ liệu ảnh nhị phân từ MongoDB
    image_data = plate_record["violation_image"]

    # Chuyển dữ liệu nhị phân thành mảng NumPy
    nparr = np.frombuffer(image_data, np.uint8)

    # Giải mã ảnh và hiển thị bằng OpenCV
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is not None:
        cv2.imshow("Ảnh biển số từ MongoDB", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Không thể giải mã ảnh!")
else:
    print("Không tìm thấy ảnh trong database!")
