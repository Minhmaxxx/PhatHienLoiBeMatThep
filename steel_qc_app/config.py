# ===================== CONFIG =====================
# lưu các tham số cấu hình (IMG_SIZE, SAVE_DIR, ALPHA_THR,...)
CKPT_PATH = "./outputs_cls_cam/mobilenetv2_best.pth"
MAP_PATH  = "./outputs_cls_cam/idx_to_class.json"
IMG_SIZE  = 224
SAVE_DIR  = "./qc_outputs"

SCENE_DELTA_THR = 6.0

# CAM/mask
PCT_START = 85
PCT_MIN   = 74
PCT_STEP  = 3
MIN_AREA  = 80

# Ngưỡng quyết định lỗi theo xác suất
ALPHA_THR = 0.6

# Grad-CAM++ target layers cho MobileNetV2
CAM_TOP_LAYER = "features.18"
CAM_AUX_LAYER = "features.17"

# Device
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Chuẩn hóa ảnh
from torchvision import transforms
MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]
TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
