# pip install pyqt5 opencv-python torch torchvision pillow numpy

import os, sys, json, time, cv2, numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets

import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms

# ===================== THEME =====================
DARK_QSS = """
* { font-family: Segoe UI, Roboto, Arial; }
QMainWindow { background: #0f1115; }
QGroupBox {
  border: 1px solid #2a2f3a; border-radius: 10px; margin-top: 16px;
  background: #12151b; color: #d8dee9; font-weight: 600; padding-top: 10px;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #9aa5b1; }
QLabel { color: #d8dee9; }
QPushButton {
  background: #1f2430; color: #e6edf3; padding: 8px 14px; border-radius: 8px;
  border: 1px solid #2a2f3a;
}
QPushButton:hover { background: #283040; }
QPushButton:pressed { background: #1a1f29; }
QComboBox, QLineEdit {
  background: #151923; color: #d8dee9; padding: 6px 10px; border-radius: 8px;
  border: 1px solid #2a2f3a;
}
QStatusBar { background: #0f1115; color: #8a93a2; }
"""

# ===================== UTILS =====================
def list_cameras(max_index=10):
    """Tìm các index camera khả dụng."""
    found = []
    backends_win = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    backends_mac = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    backends_linux = [cv2.CAP_V4L2, cv2.CAP_ANY]
    if sys.platform.startswith("win"): backends = backends_win
    elif sys.platform == "darwin":     backends = backends_mac
    else:                              backends = backends_linux
    for i in range(max_index):
        ok = False; cap=None
        for api in backends:
            cap = cv2.VideoCapture(i, api)
            if cap.isOpened():
                ok = True; break
            if cap: cap.release()
        if ok:
            found.append(i); cap.release()
    return found or [0]

def open_video_capture(source, width=1280, height=720):
    """Mở VideoCapture robust cho camera/video file."""
    if isinstance(source, int):
        if sys.platform.startswith("win"):
            candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        elif sys.platform == "darwin":
            candidates = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        else:
            candidates = [cv2.CAP_V4L2, cv2.CAP_ANY]
        for api in candidates:
            cap = cv2.VideoCapture(source, api)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(5): cap.read()
                return cap
            else:
                cap.release()
        return cv2.VideoCapture(source)  # last resort
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def qimg_from_bgr(img):
    # PyQt5 dùng Format_RGB888 an toàn trên mọi phiên bản
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)

# ===================== CONFIG =====================
CKPT_PATH = "./outputs_cls_cam/mobilenetv2_best.pth"
MAP_PATH  = "./outputs_cls_cam/idx_to_class.json"
IMG_SIZE  = 224
SAVE_DIR  = "./qc_outputs"

SCENE_DELTA_THR = 6.0

PCT_START = 85
PCT_MIN   = 74
PCT_STEP  = 3
MIN_AREA  = 80

ALPHA_THR = 0.6

NO_SIGNAL_BRIGHT_THR = 8
NO_SIGNAL_STD_THR    = 3

CAM_TOP_LAYER = "features.18"
CAM_AUX_LAYER = "features.17"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]
TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def build_model(num_classes:int):
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_feats = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_feats, num_classes)
    return m

# ===================== Grad-CAM++ =====================
class CamPlusPlus:
    def __init__(self, model, target_layer: str):
        self.model = model.eval()
        self.fmap, self.grad = None, None
        named = dict([*model.named_modules()])
        if target_layer not in named:
            raise KeyError(f"Không thấy layer '{target_layer}'.")
        layer = named[target_layer]
        self.h1 = layer.register_forward_hook(self._save_fmap)
        self.h2 = layer.register_full_backward_hook(self._save_grad)

    def _save_fmap(self, m, i, o): self.fmap = o.detach()
    def _save_grad(self, m, gi, go): self.grad = go[0].detach()
    def remove(self): self.h1.remove(); self.h2.remove()

    def __call__(self, x, class_idx=None):
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        score = logits[0, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        assert self.fmap is not None and self.grad is not None, "CAM++: fmap/grad None"

        fmap = self.fmap[0]            # (C,h,w)
        grad = self.grad[0]            # (C,h,w)
        grad2, grad3 = grad**2, grad**3
        eps = 1e-8
        sum_grad = grad.sum(dim=(1,2), keepdim=True)
        alpha = grad2 / (2*grad2 + (fmap*grad3).sum(dim=(1,2), keepdim=True) + eps)
        weights = (alpha * F.relu(sum_grad)).sum(dim=(1,2))
        cam = F.relu((weights.view(-1,1,1) * fmap).sum(0))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)
        return cam.cpu().numpy()

# ===================== Mask/BBox =====================
def heatmap_to_mask(hm, percentile=85, min_area=80):
    thr = np.percentile(hm, percentile)
    m = (hm >= thr).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)

    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    for c in cnts:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(out, [c], -1, 255, -1)
    return out

def mask_to_bboxes(mask):
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bxs=[]
    for c in cnts:
        if cv2.contourArea(c) >= MIN_AREA:
            x,y,w,h = cv2.boundingRect(c)
            bxs.append((x,y,w,h))
    return bxs

def overlay_heatmap(bgr, hm):
    hm_uint8 = np.uint8(255 * hm)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 0.6, hm_color, 0.4, 0)

def class_name_from_map(idx_to_class, cls_idx):
    try:
        return idx_to_class[cls_idx]
    except Exception:
        return idx_to_class[str(cls_idx)]

def is_no_signal(frame_bgr) -> bool:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    m   = float(gray.mean())
    s   = float(gray.std())
    mn  = int(gray.min())
    mx  = int(gray.max())
    dyn = mx - mn
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float((edges > 0).mean())
    if ((m < 6.0 or m > 249.0) and s < 2.0 and dyn < 8 and edge_ratio < 0.001):
        return True
    if mx < 10 or mn > 245:
        return True
    return False

# ===================== Worker (QThread) =====================
class Worker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray, np.ndarray, str, bool)

    def __init__(self, source, model, idx_to_class, save_dir, parent=None):
        super().__init__(parent)
        self.source = source
        self.model  = model
        self.idx_to_class = idx_to_class
        self.save_dir = save_dir
        self.stopped = False

    def run(self):
        os.makedirs(self.save_dir, exist_ok=True)
        cap = open_video_capture(self.source)
        if not cap.isOpened():
            dummy = np.zeros((240,320,3),np.uint8)
            self.frame_ready.emit(dummy, dummy, "Không mở được nguồn video/camera", True)
            return

        cam_top = CamPlusPlus(self.model, CAM_TOP_LAYER)
        cam_aux = CamPlusPlus(self.model, CAM_AUX_LAYER)

        last_small = None
        last_right, last_status, last_ok = None, "Waiting...", True

        while not self.stopped:
            ok, frame = cap.read()
            if not ok: break
            left_view = frame.copy()

            # 0) No signal
            if is_no_signal(frame):
                if last_right is None:
                    last_right = np.zeros_like(left_view)
                self.frame_ready.emit(left_view, last_right, "No signal", True)
                QtCore.QThread.msleep(10)
                continue

            # 1) Phát hiện bề mặt mới (MAD)
            small = cv2.resize(frame, (240,135))
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray  = cv2.GaussianBlur(gray, (5,5), 0)
            is_new = (last_small is None) or (float(np.mean(cv2.absdiff(gray, last_small))) > SCENE_DELTA_THR)

            if not is_new:
                if last_right is None:
                    last_right = np.zeros_like(left_view)
                self.frame_ready.emit(left_view, last_right, last_status, last_ok)
                QtCore.QThread.msleep(10)
                continue
            last_small = gray

            # 2) Suy luận
            x = TF(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = self.model(x)
                probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

            cls_idx   = int(np.argmax(probs))
            cls_name  = class_name_from_map(self.idx_to_class, cls_idx)
            max_conf  = float(probs[cls_idx])

            # 3) Quy tắc α
            H, W = frame.shape[:2]
            right_view = frame.copy()
            if max_conf < ALPHA_THR:
                status, is_ok = "OK", True
            else:
                # 4) CAM
                def try_cam(camer):
                    cam = camer(x)
                    hm  = cv2.resize(cam, (W, H), interpolation=cv2.INTER_CUBIC)
                    pct = PCT_START
                    while pct >= PCT_MIN:
                        mask = heatmap_to_mask(hm, percentile=pct, min_area=MIN_AREA)
                        bxs  = mask_to_bboxes(mask)
                        if len(bxs) > 0: return bxs, hm
                        pct -= PCT_STEP
                    return [], hm

                bboxes, hm = try_cam(cam_top)
                if len(bboxes) == 0:
                    bboxes, hm = try_cam(cam_aux)

                if len(bboxes) > 0:
                    for (x1,y1,w,h) in bboxes:
                        cv2.rectangle(right_view, (x1,y1), (x1+w,y1+h), (0,255,0), 2)
                else:
                    right_view = overlay_heatmap(right_view, hm)

                status = f"✖ {cls_name} ({max_conf:.2f})"
                is_ok  = False

            # 5) Lưu kết quả mỗi bề mặt mới
            stamp = time.strftime("%Y%m%d-%H%M%S")
            if is_ok:
                cv2.imwrite(os.path.join(self.save_dir, f"{stamp}_OK_raw.jpg"), frame)
                cv2.imwrite(os.path.join(self.save_dir, f"{stamp}_OK_view.jpg"), right_view)
            else:
                cv2.imwrite(os.path.join(self.save_dir, f"{stamp}_{cls_name}_raw.jpg"), frame)
                cv2.imwrite(os.path.join(self.save_dir, f"{stamp}_{cls_name}_view.jpg"), right_view)

            last_right, last_status, last_ok = right_view, status, is_ok
            self.frame_ready.emit(left_view, right_view, status, is_ok)
            QtCore.QThread.msleep(10)

        cap.release()

    def stop(self):
        self.stopped = True
        self.wait(500)

# ===================== MainWindow (GUI) =====================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Steel Surface QC")
        self.resize(1320, 860)

        QtWidgets.QApplication.instance().setStyleSheet(DARK_QSS)

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central); outer.setContentsMargins(14, 12, 14, 12); outer.setSpacing(12)

        # ==== Controls ====
        top_box = QtWidgets.QGroupBox("Input & Save")
        top = QtWidgets.QHBoxLayout(top_box); top.setSpacing(8)

        cams = list_cameras(10)
        self.cmb_cam = QtWidgets.QComboBox(); self.cmb_cam.addItems([str(i) for i in cams]); self.cmb_cam.setMinimumWidth(70)

        self.btn_cam = QtWidgets.QPushButton("Use Camera")
        self.btn_vid = QtWidgets.QPushButton("Open Video…")
        self.btn_img = QtWidgets.QPushButton("Open Image…")
        self.btn_dir = QtWidgets.QPushButton("Select Save Folder…")
        self.lbl_dir = QtWidgets.QLabel(os.path.abspath(SAVE_DIR))
        self.lbl_dir.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.lbl_dir.setStyleSheet("color:#9aa5b1;")

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop  = QtWidgets.QPushButton("Stop"); self.btn_stop.setEnabled(False)

        top.addWidget(QtWidgets.QLabel("Camera:")); top.addWidget(self.cmb_cam); top.addWidget(self.btn_cam)
        top.addSpacing(10); top.addWidget(self.btn_vid); top.addWidget(self.btn_img)
        top.addSpacing(10); top.addWidget(self.btn_dir); top.addWidget(self.lbl_dir, 1)
        top.addSpacing(10); top.addWidget(self.btn_start); top.addWidget(self.btn_stop)
        outer.addWidget(top_box)

        # ==== Views ====
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.setStyleSheet("QSplitter::handle { background:#1a1f29; }")

        left_frame  = QtWidgets.QFrame();  left_frame.setStyleSheet("background:#0b0d12; border-radius:12px;")
        right_frame = QtWidgets.QFrame(); right_frame.setStyleSheet("background:#0b0d12; border-radius:12px;")
        for fr in (left_frame, right_frame): fr.setFrameShape(QtWidgets.QFrame.StyledPanel)

        layL = QtWidgets.QVBoxLayout(left_frame);  layL.setContentsMargins(10,10,10,10)
        layR = QtWidgets.QVBoxLayout(right_frame); layR.setContentsMargins(10,10,10,10)

        capL = QtWidgets.QLabel("Source"); capL.setStyleSheet("color:#9aa5b1; font-size:13px;")
        capR = QtWidgets.QLabel("Result View"); capR.setStyleSheet("color:#9aa5b1; font-size:13px;")
        self.lbl_left  = QtWidgets.QLabel(); self.lbl_left.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_right = QtWidgets.QLabel(); self.lbl_right.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_left.setMinimumSize(640, 420); self.lbl_right.setMinimumSize(640, 420)
        self.lbl_left.setStyleSheet("background:#111418; border:1px solid #242a33; border-radius:10px;")
        self.lbl_right.setStyleSheet("background:#111418; border:1px solid #242a33; border-radius:10px;")

        layL.addWidget(capL); layL.addWidget(self.lbl_left, 1)
        layR.addWidget(capR); layR.addWidget(self.lbl_right, 1)

        split.addWidget(left_frame); split.addWidget(right_frame); split.setSizes([660, 660])
        outer.addWidget(split, 1)

        # ==== Status ====
        status_box = QtWidgets.QFrame(); status_box.setStyleSheet("background:#0b0d12; border-radius:14px;")
        sb = QtWidgets.QVBoxLayout(status_box); sb.setContentsMargins(14,14,14,14)

        self.lbl_status = QtWidgets.QLabel("Waiting…")
        self.lbl_status.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_status.setMinimumHeight(120)
        self.lbl_status.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #132a13, stop:1 #0f2210);"
            "border:1px solid #1f4022; border-radius:16px; color:#eaffea; font-size:34px; font-weight:700;")
        sb.addWidget(self.lbl_status)
        outer.addWidget(status_box)

        # status bar
        self.status = QtWidgets.QStatusBar(); self.setStatusBar(self.status)
        dev = (torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU')
        self.status.showMessage(f"Device: {dev}   |   Model: MobileNetV2 + Grad-CAM++   |   α={ALPHA_THR}")

        # events
        self.btn_cam.clicked.connect(self.use_camera)
        self.btn_vid.clicked.connect(self.open_video)
        self.btn_img.clicked.connect(self.open_image)
        self.btn_dir.clicked.connect(self.pick_save_dir)
        self.btn_start.clicked.connect(self.start_stream)
        self.btn_stop.clicked.connect(self.stop_stream)

        # state
        self.source = (int(cams[0]) if cams else 0)
        self.save_dir = os.path.abspath(SAVE_DIR)
        self.idx_to_class, self.model = self.load_model()
        self.cam_top = CamPlusPlus(self.model, CAM_TOP_LAYER)
        self.cam_aux = CamPlusPlus(self.model, CAM_AUX_LAYER)
        self.worker = None

    # ---------- helpers ----------
    def load_model(self):
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            idx_to_class = json.load(f)
        num_classes = len(idx_to_class)
        model = build_model(num_classes).to(DEVICE).eval()

        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        state = ckpt.get("model", ckpt)
        new_state = {}
        for k,v in state.items():
            if k.startswith("module."): k = k[len("module."):]
            new_state[k] = v
        model.load_state_dict(new_state, strict=False)
        return idx_to_class, model

    def _show_on_label(self, lbl, img_bgr):
        h,w = img_bgr.shape[:2]
        maxw, maxh = lbl.width(), lbl.height()
        s = min(maxw/max(1,w), maxh/max(1,h))
        disp = cv2.resize(img_bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        lbl.setPixmap(QtGui.QPixmap.fromImage(qimg_from_bgr(disp)))

    def _update_status(self, status_text, is_ok):
        if status_text == "No signal":
            self.lbl_status.setStyleSheet(
                "background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #2b2f3a, stop:1 #1b1f28);"
                "border:1px solid #3b4252; border-radius:16px; color:#e5e9f0; font-size:34px; font-weight:700;")
            self.lbl_status.setText("No signal")
        elif is_ok:
            self.lbl_status.setStyleSheet(
                "background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #164b2f, stop:1 #0f3a23);"
                "border:1px solid #246c45; border-radius:16px; color:#eaffea; font-size:34px; font-weight:800;")
            self.lbl_status.setText("✔ OK")
        else:
            self.lbl_status.setStyleSheet(
                "background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #5a1b1b, stop:1 #3d1111);"
                "border:1px solid #8b1e1e; border-radius:16px; color:#ffecec; font-size:34px; font-weight:800;")
            self.lbl_status.setText(status_text)

    # ---------- actions ----------
    def use_camera(self):
        self.source = int(self.cmb_cam.currentText())
        self.status.showMessage(f"Camera {self.source}")

    def open_video(self):
        fp, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open video", "", "Video files (*.mp4 *.avi *.mov *.mkv)")
        if fp:
            self.source = fp
            self.status.showMessage(f"Video: {fp}")

    def open_image(self):
        fp, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open image", "", "Image files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not fp: return
        img = cv2.imread(fp)
        if img is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Không đọc được ảnh.")
            return

        self._show_on_label(self.lbl_left, img)

        if is_no_signal(img):
            right = np.zeros_like(img)
            self._show_on_label(self.lbl_right, right)
            self._update_status("No signal", True)
            return

        right, status_text, is_ok = self.infer_on_frame(img)

        self._show_on_label(self.lbl_right, right)
        self._update_status(status_text, is_ok)

        os.makedirs(self.save_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        if is_ok:
            cv2.imwrite(os.path.join(self.save_dir, f"{stamp}_IMG_OK_raw.jpg"), img)
            cv2.imwrite(os.path.join(self.save_dir, f"{stamp}_IMG_OK_view.jpg"), right)
        else:
            cls_name = status_text.replace("✖", "").strip().split()[0]
            cv2.imwrite(os.path.join(self.save_dir, f"{stamp}_IMG_{cls_name}_raw.jpg"), img)
            cv2.imwrite(os.path.join(self.save_dir, f"{stamp}_IMG_{cls_name}_view.jpg"), right)

        self.status.showMessage(f"Predicted image: {fp}")

    def pick_save_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select save folder", self.save_dir)
        if d:
            self.save_dir = d
            self.lbl_dir.setText(d)

    def start_stream(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.worker = Worker(self.source, self.model, self.idx_to_class, self.save_dir)
        self.worker.frame_ready.connect(self.update_frames)
        self.worker.start()
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self.status.showMessage(f"Streaming: {self.source}")

    def stop_stream(self):
        if self.worker:
            self.worker.stop(); self.worker = None
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.status.showMessage("Stopped.")

    @QtCore.pyqtSlot(np.ndarray, np.ndarray, str, bool)
    def update_frames(self, left_bgr, right_bgr, status_text, is_ok):
        self._show_on_label(self.lbl_left, left_bgr)
        self._show_on_label(self.lbl_right, right_bgr)
        self._update_status(status_text, is_ok)

    def infer_on_frame(self, frame_bgr):
        """Ảnh đơn: trả (view, status_text, is_ok) theo ngưỡng α + CAM."""
        x = TF(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        cls_idx  = int(np.argmax(probs))
        cls_name = class_name_from_map(self.idx_to_class, cls_idx)
        max_conf = float(probs[cls_idx])

        view = frame_bgr.copy()
        if max_conf < ALPHA_THR:
            return view, "OK", True

        H, W = frame_bgr.shape[:2]
        def try_cam(cam_obj):
            cam = cam_obj(x)
            hm  = cv2.resize(cam, (W, H), interpolation=cv2.INTER_CUBIC)
            pct = PCT_START
            while pct >= PCT_MIN:
                mask = heatmap_to_mask(hm, percentile=pct, min_area=MIN_AREA)
                bxs  = mask_to_bboxes(mask)
                if len(bxs) > 0:
                    return bxs, hm
                pct -= PCT_STEP
            return [], hm

        bboxes, hm = try_cam(self.cam_top)
        if len(bboxes) == 0:
            bboxes, hm = try_cam(self.cam_aux)

        if len(bboxes) > 0:
            for (x1,y1,w,h) in bboxes:
                cv2.rectangle(view, (x1,y1), (x1+w,y1+h), (0,255,0), 2)
        else:
            view = overlay_heatmap(view, hm)

        return view, f"✖ {cls_name} ({max_conf:.2f})", False

    def closeEvent(self, e):
        self.stop_stream()
        try:
            self.cam_top.remove(); self.cam_aux.remove()
        except Exception:
            pass
        e.accept()

# ===================== ENTRY =====================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())
