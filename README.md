# 🔩 Metal Surface Defect Detection using Deep Learning (CNN)

Hệ thống phát hiện lỗi bề mặt kim loại trong dây chuyền sản xuất công nghiệp, sử dụng **ResNet50** (PyTorch) kết hợp **Grad-CAM++** để trực quan hóa vùng lỗi và **WeightedRandomSampler** xử lý mất cân bằng dữ liệu.

> Đồ án chuyên ngành – Ngành Khoa học Máy tính  
> Trường Đại học Công Nghiệp Hà Nội – 2025  

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Dataset](#dataset)
- [Kiến trúc mô hình](#kiến-trúc-mô-hình)
- [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
- [Cài đặt](#cài-đặt)
- [Cách sử dụng](#cách-sử-dụng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)

---

## Tổng quan

Trong sản xuất công nghiệp, kiểm tra chất lượng bề mặt kim loại bằng mắt thường tốn thời gian, dễ sai sót và không thể hoạt động liên tục 24/7. Dự án này xây dựng một hệ thống tự động:

- **Phân loại** ảnh bề mặt kim loại thành 10 loại lỗi
- **Xử lý mất cân bằng dữ liệu** bằng WeightedRandomSampler và class weights
- **Trực quan hóa** vùng lỗi bằng Grad-CAM++ (heatmap + bounding contour)
- **Phát hiện thời gian thực** qua webcam hoặc video
- **Giao diện desktop** (PyQt5) dễ sử dụng cho nhân viên QC

## Dataset

**Bộ dữ liệu: GC10-DET (Garbage Classification 10 – Defect Detection)**  
Dataset công nghiệp gồm ảnh bề mặt kim loại với 10 loại khuyết tật thực tế.

>  Tải dataset tại: [Kaggle – GC10-DET](https://www.kaggle.com/datasets/alex000kim/gc10det)

### Các lớp lỗi

| STT | Lớp lỗi | Mô tả | Số mẫu gốc |
|-----|---------|-------|------------|
| 1 | `crease` | Nếp gấp trên bề mặt | 50 |
| 2 | `crescent_gap` | Khe hình lưỡi liềm | 204 |
| 3 | `inclusion` | Tạp chất lẫn trong kim loại | 185 |
| 4 | `oil_spot` | Vết dầu loang | 225 |
| 5 | `punching_hole` | Lỗ đột không mong muốn | 223 |
| 6 | `rolled_pit` | Lõm do cán ép | 35 |
| 7 | `silk_spot` | Vết sợi mảnh kéo dài | 655 |
| 8 | `waist_folding` | Gập eo trên dải kim loại | 150 |
| 9 | `water_spot` | Vết nước để lại | 292 |
| 10 | `welding_line` | Đường hàn không đều | 293 |

**Tổng: ~2.312 ảnh** · Định dạng: `.jpg` / `.png` · Input size: **384×384**

<img width="778" height="400" alt="image" src="https://github.com/user-attachments/assets/a74c7d4e-0b5c-49e8-93b6-60e512d5e8ec" />


### Xử lý mất cân bằng dữ liệu

Chênh lệch lớn giữa `silk_spot` (655) và `rolled_pit` (35), nhóm áp dụng:

- **Data Augmentation**: ColorJitter, RandomHorizontalFlip, Resize — áp dụng khi training
- **WeightedRandomSampler** (PyTorch): tăng tần suất xuất hiện lớp hiếm trong mỗi batch
- Chuẩn hóa pixel theo **ImageNet mean/std**: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

Mỗi lớp được nâng lên ~**1.200 mẫu** sau augmentation.


### Tỷ lệ chia dữ liệu

| Tập | Tỷ lệ | Nguồn |
|-----|-------|-------|
| Train | 70% | Gốc + augmented |
| Validation | 15% | Chỉ dữ liệu gốc |
| Test | 15% | Chỉ dữ liệu gốc |

---

## Kiến trúc mô hình

### ResNet50 (Pretrained ImageNet)

<img width="622" height="329" alt="image" src="https://github.com/user-attachments/assets/22b595dd-efa8-4c61-8bad-5cd200cee587" />


Mô hình sử dụng **ResNet50 pretrained `IMAGENET1K_V2`**, thay thế fully-connected layer cuối:

```
ResNet50 backbone (pretrained)
    └── fc: Linear(2048 → 10)   # 10 lớp lỗi
```

### Hyperparameters

| Tham số | Giá trị |
|---------|---------|
| Input size | 384×384×3 |
| Batch size | 16 |
| Epochs | 8 |
| Learning rate | 1e-4 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=8) |
| Loss | CrossEntropyLoss |
| Sampler | WeightedRandomSampler |

### Grad-CAM++

<img width="1174" height="841" alt="image" src="https://github.com/user-attachments/assets/976c0446-7d6b-49f4-9513-bfe7aa863232" />


Kỹ thuật nâng cấp từ Grad-CAM, tính toán trên **layer4** của ResNet50:

1. Tính gradient bậc 2 và bậc 3 để xác định **alpha weights** chính xác hơn
2. Sinh **heatmap** từ tổ hợp tuyến tính của feature maps có trọng số
3. Ngưỡng vùng nóng ở **percentile 85** + morphological opening/closing để làm mượt
4. Vẽ **contour** quanh vùng lỗi (loại bỏ contour < 80px²)

---

## Kết quả thực nghiệm

<img width="1156" height="741" alt="image" src="https://github.com/user-attachments/assets/6c3e0e7a-2bc6-4e02-8e21-d634a541e5b2" />


<img width="1157" height="408" alt="image" src="https://github.com/user-attachments/assets/718fc939-e3b1-4b18-9047-434691e90e52" />


<img width="779" height="689" alt="image" src="https://github.com/user-attachments/assets/a506657a-04d7-43b2-b572-0f1ce96c9b89" />


<img width="783" height="682" alt="image" src="https://github.com/user-attachments/assets/2efe4302-5f89-41d7-98ab-ea73b5ed616f" />


### Classification Report (tập Test)

<img width="574" height="390" alt="image" src="https://github.com/user-attachments/assets/a5958b81-0775-4634-a692-6e821c3bef70" />

---

##  Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- GPU khuyến nghị (CUDA 11+), CPU chạy được nhưng chậm hơn nhiều

### Cài đặt thư viện

```bash
git clone https://github.com/<your-username>/metal-defect-detection.git
cd metal-defect-detection
pip install -r requirements.txt
```

### Tổ chức dataset

Sau khi tải GC10-DET, tổ chức theo cấu trúc `ImageFolder` của PyTorch:

```
data_train/
├── train/
│   ├── crease/
│   ├── crescent_gap/
│   ├── inclusion/
│   ├── oil_spot/
│   ├── punching_hole/
│   ├── rolled_pit/
│   ├── silk_spot/
│   ├── waist_folding/
│   ├── water_spot/
│   └── welding_line/
├── val/
│   └── (cấu trúc tương tự)
└── test/
    └── (cấu trúc tương tự)
```

Sau đó cập nhật đường dẫn trong notebook:
```python
DATA_ROOT = "đường/dẫn/tới/data_train"
```

---

## Cách sử dụng

### 1. Chạy notebook huấn luyện

Mở `main.ipynb` trên **Google Colab** hoặc **Jupyter Notebook**, chạy tuần tự theo các section:

| Section | Mô tả |
|---------|-------|
| Cài gói & cấu hình | Import thư viện, set hyperparameters |
| Đọc dữ liệu & thống kê | Load ImageFolder, vẽ biểu đồ phân bố lớp |
| DataLoader | Tạo WeightedRandomSampler, train/val/test loader |
| Huấn luyện ResNet50 | Fine-tune + CosineAnnealingLR, lưu best model |
| Đánh giá chi tiết | Loss/Accuracy curve, Confusion Matrix, Classification Report |
| Grad-CAM++ | Trực quan hóa vùng lỗi trên ảnh test |

### 2. Test Grad-CAM++ với ảnh tùy chỉnh

```python
visualize_one("đường/dẫn/ảnh.jpg", percentile=85)
```

Output gồm 3 cột: **Ảnh gốc** | **Overlay heatmap** | **Contour vùng lỗi**

### 3. Chạy ứng dụng desktop

```bash
python gui_app/main.py
```
<img width="945" height="530" alt="image" src="https://github.com/user-attachments/assets/d5ef1ef3-5858-4791-baee-9b8fb182eb5a" />


Trong giao diện:
- Chọn nguồn đầu vào: **Ảnh / Video / Camera**
- Chọn thư mục lưu kết quả
- Nhấn **Start** để bắt đầu phát hiện
- Kết quả hiển thị trực tiếp kèm Grad-CAM++ heatmap

---

##  Cấu trúc dự án

```
metal-defect-detection/
│
├── main.ipynb                     # Notebook chính: training, evaluation, Grad-CAM++
│
├── data_train/                    # Dataset GC10-DET đã chia (train/val/test)
│
├── outputs_cls_cam/               # Output sau training
│   ├── resnet50_cls.pth           # Best model weights
│   └── idx_to_class.json          # Mapping index → tên lớp
│
├── gui_app/                       # Ứng dụng desktop PyQt5
│
├── docs/
│   └── images/                    # Ảnh minh họa dùng trong README
│
├── requirements.txt               # Thư viện Python
└── README.md
```

---

##  Tài liệu tham khảo

- He et al. (2015) – *Deep Residual Learning for Image Recognition* (ResNet)
- Chattopadhay et al. (2018) – *Grad-CAM++: Generalized Gradient-based Visual Explanations*
- Dataset: GC10-DET – Industrial surface defect detection dataset

---

## 📄 Giấy phép

Dự án phục vụ mục đích học thuật. Vui lòng trích dẫn nếu sử dụng lại.
