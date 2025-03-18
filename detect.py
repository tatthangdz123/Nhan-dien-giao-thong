import cv2
import os

# Đường dẫn đến video
video_path = "5P.mp4"

# Tạo thư mục 'frames' để lưu ảnh nếu chưa có
output_folder = "frames-2"
os.makedirs(output_folder, exist_ok=True)

# Mở video bằng OpenCV
cap = cv2.VideoCapture(video_path)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Dừng khi hết video

    # Lưu frame vào thư mục 'frames' dưới dạng ảnh JPG
    frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_path, frame)

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"✅ Đã tách {frame_count} frames và lưu vào thư mục '{output_folder}'")
