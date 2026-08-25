import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models
import numpy as np
# build_model(), load checkpoint, Grad-CAM++
def build_model(num_classes:int):
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_feats = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_feats, num_classes)
    return m

class CamPlusPlus:
    def __init__(self, model, target_layer: str):
        self.model = model.eval()
        self.fmap, self.grad = None, None
        named = dict([*model.named_modules()])
        if target_layer not in named:
            raise KeyError(f"Không thấy layer '{target_layer}'")
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

        fmap = self.fmap[0]
        grad = self.grad[0]
        grad2, grad3 = grad**2, grad**3
        eps = 1e-8
        sum_grad = grad.sum(dim=(1,2), keepdim=True)
        alpha = grad2 / (2*grad2 + (fmap*grad3).sum(dim=(1,2), keepdim=True) + eps)
        weights = (alpha * F.relu(sum_grad)).sum(dim=(1,2))
        cam = F.relu((weights.view(-1,1,1) * fmap).sum(0))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)
        return cam.cpu().numpy()

def heatmap_to_mask(hm, percentile=85, min_area=80):
    import cv2
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

def mask_to_bboxes(mask, min_area=80):
    import cv2
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bxs=[]
    for c in cnts:
        if cv2.contourArea(c) >= min_area:
            x,y,w,h = cv2.boundingRect(c)
            bxs.append((x,y,w,h))
    return bxs

def overlay_heatmap(bgr, hm):
    import cv2
    hm_uint8 = np.uint8(255 * hm)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 0.6, hm_color, 0.4, 0)

def class_name_from_map(idx_to_class, cls_idx):
    try:
        return idx_to_class[cls_idx]
    except Exception:
        return idx_to_class[str(cls_idx)]
