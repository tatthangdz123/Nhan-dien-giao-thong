# Cài đặt các thư viện cần thiết
from IPython.display import display, Image
import os
import glob
from ultralytics import YOLO
from roboflow import Roboflow

# Kiểm tra môi trường YOLOv8
import ultralytics
ultralytics.checks()

# Load mô hình YOLOv8
model = YOLO("yolov8n.pt")

# Dự đoán trên ảnh mẫu sử dụng YOLOv8
results = model.predict(source="https://media.roboflow.com/notebooks/examples/dog.jpeg", conf=0.25, save=True)

# Kiểm tra đường dẫn thực tế trong thư mục runs/detect/predict
prediction_dir = os.path.join("runs", "detect", "predict")
if os.path.exists(prediction_dir):
    image_paths = glob.glob(f"{prediction_dir}/*.jpg")
    if image_paths:
        display(Image(filename=image_paths[0], height=600))

# In thông tin kết quả dự đoán
print("Toạ độ hộp giới hạn:", results[0].boxes.xyxy)  # Toạ độ hộp giới hạn
print("Độ tin cậy:", results[0].boxes.conf)  # Độ tin cậy
print("Nhãn lớp:", results[0].boxes.cls)   # Nhãn lớp

# Tải dataset từ Roboflow
rf = Roboflow(api_key="C9JqykrzPzqftRLusM9a")  # Thay API key của bạn nếu cần
project = rf.workspace("thang-azynm").project("lisence_plate-kwxxx")
version = project.version(1)
dataset = version.download("yolov8")

# Huấn luyện mô hình YOLOv8 sử dụng dataset từ Roboflow
model = YOLO("yolov8s.pt")
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=25,
    imgsz=800,
    plots=True
)

# Hiển thị hình ảnh kết quả huấn luyện
train_results = [
    os.path.join("runs", "detect", "train", "confusion_matrix.png"),
    os.path.join("runs", "detect", "train", "results.png"),
    os.path.join("runs", "detect", "train", "val_batch0_pred.jpg")
]

for img in train_results:
    if os.path.exists(img):
        display(Image(filename=img, width=600))
    else:
        print(f"Không tìm thấy {img}")

# Đánh giá mô hình sau khi huấn luyện
best_model_path = os.path.join("runs", "detect", "train", "weights", "best.pt")
best_model = YOLO(best_model_path)
metrics = best_model.val(data=f"{dataset.location}/data.yaml")

# Hiển thị độ chính xác (mAP) và các chỉ số đánh giá
print(f"mAP50: {metrics.box.map50:.4f}")  # Độ chính xác trung bình @ IoU 0.5
print(f"mAP50-95: {metrics.box.map:.4f}")  # Độ chính xác trung bình @ IoU 0.5-0.95
print(f"Precision: {metrics.box.precision.mean():.4f}")  # Độ chính xác (Precision)
print(f"Recall: {metrics.box.recall.mean():.4f}")  # Độ nhạy (Recall)

# Dự đoán với mô hình đã huấn luyện
results = best_model.predict(
    source=os.path.join(dataset.location, "test", "images"),
    conf=0.25,
    save=True
)

# Hiển thị hình ảnh dự đoán
base_path = os.path.join("runs", "detect")
subfolders = [os.path.join(base_path, d) for d in os.listdir(base_path)
              if os.path.isdir(os.path.join(base_path, d)) and d.startswith('predict')]

if subfolders:
    latest_folder = max(subfolders, key=os.path.getmtime)
    image_paths = glob.glob(f"{latest_folder}/*.jpg")[:3]

    for image_path in image_paths:
        display(Image(filename=image_path, width=600))
        print("\n")
else:
    print("Không tìm thấy thư mục dự đoán!")
