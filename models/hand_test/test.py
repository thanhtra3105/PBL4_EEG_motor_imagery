import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os

# --- CẤU HÌNH ---
# Đảm bảo Python nhìn thấy folder chứa code model
sys.path.append(os.getcwd()) 

try:
    from model_set.models import HopefullNet
except ImportError:
    print("❌ Lỗi: Không tìm thấy folder 'model_set'. Hãy để file này ngang hàng với folder model_set.")
    sys.exit(1)

# --- LOAD DỮ LIỆU & MODEL ---
print("1. Đang load dữ liệu Test từ file .npy...")
try:
    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')
    print(f"   -> Đã load {len(x_test)} mẫu dữ liệu.")
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file x_test.npy hoặc y_test.npy")
    sys.exit(1)

print("2. Đang khởi tạo Model...")
model = HopefullNet()
# Chạy mồi 1 mẫu để build model
_ = model(np.zeros((1, 640, 2))) 

print("3. Đang nạp trọng số (Weights)...")
model.load_weights('bestModel.h5')

# --- CHẠY TEST ---
print("\n=== KẾT QUẢ CHẠY TRÊN MÁY LOCAL ===")
# Compile để tính accuracy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"🏆 Độ chính xác: {acc*100:.2f}%")

# Dự đoán
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))