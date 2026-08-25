import os
import cv2
import glob
import random
from collections import deque
from pathlib import Path
from typing import List, Dict, Deque, Tuple

def scan_dataset(root: str, exclude_ok: bool = False) -> Dict[str, Deque[str]]:
    """
    Scan dataset organized as: root/class_name/*.jpg|png
    Returns dict[class_name] -> deque of image paths (shuffled).
    """
    rootp = Path(root)
    classes: Dict[str, Deque[str]] = {}
    for sub in sorted([p for p in rootp.iterdir() if p.is_dir()]):
        cname = sub.name
        if exclude_ok and cname.lower() in {"ok", "normal", "no_defect", "none"}:
            continue  # handle OK separately
        imgs: List[str] = []
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp"):
            imgs.extend(glob.glob(str(sub / ext)))
        if not imgs:
            continue
        random.shuffle(imgs)
        classes[cname] = deque(imgs)
    return classes

def load_ok_images(ok_folder: str) -> Deque[str]:
    if not ok_folder:
        return deque()
    okp = Path(ok_folder)
    if not okp.exists():
        return deque()
    imgs: List[str] = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp"):
        imgs.extend(glob.glob(str(okp / ext)))
    random.shuffle(imgs)
    return deque(imgs)

def synthesize_ok_frame(base_shape: Tuple[int,int]):
    import numpy as np
    h, w = base_shape
    # nền xám trung tính (giống bề mặt thép)
    base_gray = 128  # xám trung bình thay vì trắng sáng
    img = np.full((h, w, 3), base_gray, dtype=np.uint8)

    # thêm noise Gaussian nhẹ để không quá phẳng
    noise = np.random.normal(0, 8, (h, w, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def read_image(path: str, target_size: Tuple[int, int]=None):
    img = cv2.imread(path)
    if img is None:
        return None
    if target_size is not None:
        img = cv2.resize(img, target_size)
    return img

# (Không dùng nữa) — để lại nếu cần tham khảo
def overlay_label(img, text: str):
    if not text:
        return img
    out = img.copy()
    cv2.rectangle(out, (0,0), (out.shape[1], 36), (0,0,0), -1)
    cv2.putText(out, text, (10,26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    return out

def plan_round_robin(classes: Dict[str, Deque[str]],
                     max_images_per_class: int = None,
                     max_same_in_a_row: int = 1) -> List[Tuple[str,str]]:
    """
    Build playlist: list of (class_name, image_path) by round-robin.
    """
    work: Dict[str, Deque[str]] = {
        c: deque(list(paths)[:max_images_per_class] if max_images_per_class else list(paths))
        for c, paths in classes.items() if paths
    }
    order: List[Tuple[str,str]] = []
    last_class = None
    same_count = 0
    while any(work.values()):
        candidates = [c for c, dq in work.items() if dq]
        if not candidates:
            break
        random.shuffle(candidates)
        if last_class is None or same_count < max_same_in_a_row:
            pick = candidates[0]
        else:
            diffs = [c for c in candidates if c != last_class]
            pick = diffs[0] if diffs else candidates[0]
        path = work[pick].popleft()
        order.append((pick, path))
        if pick == last_class:
            same_count += 1
        else:
            last_class = pick
            same_count = 1
    return order

def insert_ok_every(order: List[Tuple[str,str]],
                    every_n: int,
                    ok_imgs: Deque[str]) -> List[Tuple[str,str]]:
    """
    Insert ('OK', path_or_None) every N items.
    If ok_imgs empty, None means synthesize OK frame later.
    """
    if every_n <= 0:
        return order
    result: List[Tuple[str,str]] = []
    count = 0
    for item in order:
        result.append(item)
        count += 1
        if count % every_n == 0:
            if ok_imgs:
                result.append(("OK", ok_imgs[0]))
                ok_imgs.rotate(-1)
            else:
                result.append(("OK", None))
    return result

def write_demo_video(playlist: List[Tuple[str,str]],
                     out_path: str,
                     seconds_per_image: float = 5.0,
                     fps: int = 25,
                     spacer_frames: int = 2,
                     frame_size: Tuple[int,int] = None,
                     label: bool = False):  # label luôn False để không vẽ chữ
    """
    Write video: show each image for seconds_per_image, insert black spacer frames
    so MAD spikes between images. Absolutely no text overlay when label=False.
    """
    if not playlist:
        raise ValueError("Empty playlist")

    first_img_path = next((p for _, p in playlist if p is not None), None)
    if first_img_path is None:
        frame_size = frame_size or (640, 480)
    else:
        img0 = cv2.imread(first_img_path)
        if img0 is None:
            raise RuntimeError(f"Cannot read first image: {first_img_path}")
        h0, w0 = img0.shape[:2]
        if frame_size is None:
            frame_size = (w0, h0)

    w, h = frame_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    frames_per_image = max(1, int(round(seconds_per_image * fps)))

    black = None
    for cls_name, path in playlist:
        if path is None:
            img = synthesize_ok_frame((h, w))
        else:
            img = read_image(path, (w, h))
            if img is None:
                continue

        # KHÔNG vẽ bất kỳ chữ nào:
        # if label: ...  # <- đã tắt

        for _ in range(frames_per_image):
            writer.write(img)

        if spacer_frames > 0:
            if black is None:
                black = np.zeros_like(img)
            for _ in range(spacer_frames):
                writer.write(black)

    writer.release()
    print(f"[OK] Saved video: {out_path} (fps={fps}, per_image={seconds_per_image}s, spacer={spacer_frames})")

def build_demo_video(root_dataset: str,
                     out_path: str = "demo_video.mp4",
                     ok_folder: str = "",
                     max_images_per_class: int = 20,
                     every_n_insert_ok: int = 3,
                     max_same_in_a_row: int = 1,
                     seconds_per_image: float = 5.0,
                     fps: int = 25,
                     spacer_frames: int = 2,
                     frame_size: Tuple[int,int] = None,
                     seed: int = 42):
    """
    High-level helper:
    - round-robin across classes
    - insert OK every N samples (from ok_folder if provided, else synthesize)
    - write video with spacer frames to trigger MAD between images
    """
    random.seed(seed)
    classes = scan_dataset(root_dataset, exclude_ok=True)
    ok_imgs = load_ok_images(ok_folder)
    order = plan_round_robin(classes,
                             max_images_per_class=max_images_per_class,
                             max_same_in_a_row=max_same_in_a_row)
    playlist = insert_ok_every(order, every_n_insert_ok, ok_imgs)
    write_demo_video(playlist, out_path,
                     seconds_per_image=seconds_per_image,
                     fps=fps,
                     spacer_frames=spacer_frames,
                     frame_size=frame_size,
                     label=False)  # <-- BẮT BUỘC: Không vẽ chữ

if __name__ == "__main__":
    import argparse
    import numpy as np  # used for black frame generation
    ap = argparse.ArgumentParser(description="Build demo video from image dataset with constraints (no overlay text).")
    ap.add_argument("--root", required=True, help="Dataset root with subfolders per class")
    ap.add_argument("--out", default="demo_video.mp4", help="Output video path")
    ap.add_argument("--ok", default="", help="Folder containing OK images (optional). If missing, OK is synthesized.")
    ap.add_argument("--max_per_class", type=int, default=20, help="Max images per class to include")
    ap.add_argument("--insert_ok_every", type=int, default=3, help="Insert an OK frame every N images (0=disable)")
    ap.add_argument("--max_same", type=int, default=1, help="Max consecutive images from same class")
    ap.add_argument("--sec_per_img", type=float, default=5.0, help="Seconds per image")
    ap.add_argument("--fps", type=int, default=25, help="Video FPS")
    ap.add_argument("--spacer", type=int, default=2, help="Black spacer frames between images (to force MAD spikes)")
    ap.add_argument("--width", type=int, default=0, help="Force output width (0 = auto)")
    ap.add_argument("--height", type=int, default=0, help="Force output height (0 = auto)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")

    args = ap.parse_args()
    size = None
    if args.width > 0 and args.height > 0:
        size = (args.width, args.height)
    build_demo_video(root_dataset=args.root,
                     out_path=args.out,
                     ok_folder=args.ok,
                     max_images_per_class=args.max_per_class,
                     every_n_insert_ok=args.insert_ok_every,
                     max_same_in_a_row=args.max_same,
                     seconds_per_image=args.sec_per_img,
                     fps=args.fps,
                     spacer_frames=args.spacer,
                     frame_size=size,
                     seed=args.seed)
