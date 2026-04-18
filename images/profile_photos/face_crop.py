"""Face-centered square crop for headshots (InsightFace + fitted ellipse).

Usage: python face_crop.py <input> <output> [--debug]

Detects the largest face, fits an ellipse to the 106 landmarks, and crops a
square centered on the ellipse with face diameter ~1/2.70 of crop size.
--debug overlays the circle mask, face hull/ellipse, eyes, and chin point.

Calibration derived from Scott Selikoff's reference headshot (600x600):
    face ellipse diameter = 222
    crop size / face diameter = 2.70
    face center = (301, 284) — very nearly co-centered with image center
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis

TARGET_SIZE = 800
CROP_TO_FACE_DIAM = 2.70  # Scott's 600/222

_app = None


def get_app():
    global _app
    if _app is None:
        _app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app


def detect(img):
    faces = get_app().get(img)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def compute_crop(img_h, img_w, face):
    pts = face.landmark_2d_106.astype(np.float32)
    (cx, cy), (ax1, ax2), angle = cv2.fitEllipse(pts)
    face_diam = max(ax1, ax2)

    # Always crop at the ideal size centered on the face. When this extends
    # beyond the source image, the overshoot is edge-replicated in
    # extract_crop so small overruns blend in without leaving black bars.
    crop_size = int(face_diam * CROP_TO_FACE_DIAM)
    top = int(cy - crop_size / 2)
    left = int(cx - crop_size / 2)

    return top, left, crop_size, (cx, cy), face_diam, (ax1, ax2, angle)


def extract_crop(img, top, left, crop_size):
    """Return a crop_size x crop_size region. Where the requested window
    extends outside the source, pad by replicating the edge pixel outwards
    (cv2.BORDER_REPLICATE) so tight crops don't leave black bars."""
    img_h, img_w = img.shape[:2]
    src_t = max(0, top)
    src_l = max(0, left)
    src_b = min(img_h, top + crop_size)
    src_r = min(img_w, left + crop_size)
    if src_t >= src_b or src_l >= src_r:
        return np.zeros((crop_size, crop_size, 3), dtype=img.dtype)

    pad_t = src_t - top
    pad_b = (top + crop_size) - src_b
    pad_l = src_l - left
    pad_r = (left + crop_size) - src_r
    region = img[src_t:src_b, src_l:src_r]
    return cv2.copyMakeBorder(region, pad_t, pad_b, pad_l, pad_r,
                              cv2.BORDER_REPLICATE)


def draw_debug(resized, top, left, crop_size, face_center, face_axes_angle, face):
    scale = TARGET_SIZE / crop_size
    cv2.circle(resized, (TARGET_SIZE // 2, TARGET_SIZE // 2),
               TARGET_SIZE // 2, (0, 255, 0), 3)

    def to_local(p):
        return int((p[0] - left) * scale), int((p[1] - top) * scale)

    pts106 = face.landmark_2d_106
    local_pts = np.array([to_local(p) for p in pts106], dtype=np.int32)
    hull = cv2.convexHull(local_pts)
    cv2.polylines(resized, [hull], isClosed=True, color=(0, 255, 255), thickness=2)

    # Fitted face ellipse, transformed to crop space.
    ax1, ax2, angle = face_axes_angle
    fc = to_local(face_center)
    cv2.ellipse(resized, fc, (int(ax1 * scale / 2), int(ax2 * scale / 2)),
                angle, 0, 360, (255, 255, 0), 2)
    cv2.drawMarker(resized, fc, (255, 255, 0),
                   markerType=cv2.MARKER_CROSS, markerSize=30, thickness=3)

    eye_l = to_local(face.kps[0])
    eye_r = to_local(face.kps[1])
    cv2.line(resized, eye_l, eye_r, (0, 200, 255), 2)
    cv2.circle(resized, eye_l, 6, (0, 0, 255), -1)
    cv2.circle(resized, eye_r, 6, (0, 0, 255), -1)


def main():
    args = sys.argv[1:]
    debug = "--debug" in args
    args = [a for a in args if a != "--debug"]
    if len(args) != 2:
        print("Usage: face_crop.py <input> <output> [--debug]", file=sys.stderr)
        sys.exit(1)

    src, dst = Path(args[0]), Path(args[1])
    img = cv2.imread(str(src))
    if img is None:
        print(f"Could not read {src}", file=sys.stderr)
        sys.exit(1)

    face = detect(img)
    if face is None:
        print(f"No face found in {src}", file=sys.stderr)
        sys.exit(1)

    h, w = img.shape[:2]
    top, left, size, face_center, face_diam, face_axes_angle = compute_crop(h, w, face)
    cropped = extract_crop(img, top, left, size)
    resized = cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

    pad_t = max(0, -top)
    pad_l = max(0, -left)
    pad_b = max(0, (top + size) - h)
    pad_r = max(0, (left + size) - w)
    pad_note = ""
    if pad_t or pad_l or pad_b or pad_r:
        pad_note = f" pad(t={pad_t},b={pad_b},l={pad_l},r={pad_r})"

    if debug:
        draw_debug(resized, top, left, size, face_center, face_axes_angle, face)

    params = []
    if dst.suffix.lower() in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 92]
    elif dst.suffix.lower() == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]

    ok = cv2.imwrite(str(dst), resized, params)
    if not ok:
        print(f"Failed to write {dst}", file=sys.stderr)
        sys.exit(1)

    face_ratio = face_diam * (TARGET_SIZE / size) / TARGET_SIZE
    print(f"{src.name}: face_diam={face_diam:.0f} crop={size} "
          f"face_ratio={face_ratio:.2f} "
          f"crop=({left},{top},{size}x{size}){pad_note} -> {dst.name}")


if __name__ == "__main__":
    main()
