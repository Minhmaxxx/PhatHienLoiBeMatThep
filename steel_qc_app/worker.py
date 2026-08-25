import os, time, cv2, numpy as np
from PyQt5 import QtCore
from PIL import Image
import torch
from utils import open_video_capture, is_no_signal
from model import CamPlusPlus, heatmap_to_mask, mask_to_bboxes, overlay_heatmap, class_name_from_map
from config import CAM_TOP_LAYER, CAM_AUX_LAYER, SCENE_DELTA_THR, PCT_START, PCT_MIN, PCT_STEP, MIN_AREA, ALPHA_THR, DEVICE, TF
# QThread xử lý video/camera
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

            # 1) Phát hiện bề mặt mới
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

            # 4) Lưu kết quả
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
