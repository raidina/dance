"""
reference_processor.py
-----------------------
ดาวน์โหลดวิดีโอ K-pop จาก YouTube และ extract pose keypoints
ด้วย MediaPipe Pose สำหรับใช้เป็น reference ในการเปรียบเทียบ

v3: เปลี่ยน person detector เป็น YOLOv8 — แม่นยำกว่า HOG มาก
    รองรับท่าเต้นผิดปกติ แขนขายก ก้มต่ำ ได้ดี
"""

import os
import cv2
import json
import numpy as np
import mediapipe as mp
import urllib.request
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision
from scipy.optimize import linear_sum_assignment

def _get_pose_model(model_dir: str = ".") -> str:
    """ดาวน์โหลด pose landmarker model ถ้ายังไม่มี"""
    path = os.path.join(model_dir, "pose_landmarker_lite.task")
    if not os.path.exists(path):
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "pose_landmarker/pose_landmarker_lite/float16/1/"
               "pose_landmarker_lite.task")
        print("กำลังดาวน์โหลด pose model (~5MB)...")
        urllib.request.urlretrieve(url, path)
        print(f"✅ ดาวน์โหลด model เสร็จ")
    return path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import subprocess
from typing import Optional, Tuple, List


def get_video_rotation(video_path: str) -> int:
    """อ่าน rotation metadata จากวิดีโอด้วย ffprobe"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_streams', video_path],
            capture_output=True, text=True
        )
        info = json.loads(result.stdout)
        for stream in info.get('streams', []):
            for sd in stream.get('side_data_list', []):
                if 'rotation' in sd:
                    return int(sd['rotation'])
            tags = stream.get('tags', {})
            if 'rotate' in tags:
                return int(tags['rotate'])
    except Exception:
        pass
    return 0


def apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """หมุนเฟรมตาม rotation metadata"""
    if rotation == 90 or rotation == -270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 270 or rotation == -90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180 or rotation == -180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DANCE_KEYPOINTS = [
    0,   # nose
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    23,  # left hip
    24,  # right hip
    25,  # left knee
    26,  # right knee
    27,  # left ankle
    28,  # right ankle
]

KEYPOINT_NAMES = {
    0: "nose", 11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
}

# สีสำหรับแสดงแต่ละคนใน preview (BGR)
PERSON_COLORS = [
    (0, 200, 255),   # เหลือง
    (0, 128, 255),   # ส้ม
    (255, 100, 0),   # น้ำเงิน
    (100, 255, 0),   # เขียว
    (255, 0, 200),   # ม่วง
    (0, 255, 200),   # เขียวอ่อน
]


# ---------------------------------------------------------------------------
# KalmanTrack + SORTTracker — multi-person tracker ที่ handle crossing ได้
# ---------------------------------------------------------------------------

class KalmanTrack:
    """
    Kalman filter สำหรับ track bounding box หนึ่งคน
    State: [cx, cy, w, h, vcx, vcy]  (ตำแหน่ง + velocity)
    """
    _id_counter = 0

    def __init__(self, bbox: Tuple[int, int, int, int]):
        KalmanTrack._id_counter += 1
        self.track_id = KalmanTrack._id_counter
        self.hits = 1
        self.no_match = 0

        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        w, h = float(bbox[2]), float(bbox[3])

        self.x = np.array([cx, cy, w, h, 0., 0.])   # state

        # state transition: constant velocity
        self.F = np.array([
            [1,0,0,0,1,0],
            [0,1,0,0,0,1],
            [0,0,1,0,0,0],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1],
        ], dtype=float)

        # measurement matrix (observe cx,cy,w,h)
        self.H = np.zeros((4, 6))
        self.H[:4, :4] = np.eye(4)

        self.P = np.diag([100., 100., 100., 100., 500., 500.])
        self.Q = np.diag([1.,   1.,   10.,  10.,  1.,   1.  ])
        self.R = np.diag([10.,  10.,  20.,  20.  ])

    def predict(self) -> Tuple[int, int, int, int]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._to_bbox()

    def update(self, bbox: Tuple[int, int, int, int]):
        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        z = np.array([cx, cy, float(bbox[2]), float(bbox[3])])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.hits += 1
        self.no_match = 0

    def _to_bbox(self) -> Tuple[int, int, int, int]:
        cx, cy, w, h = self.x[:4]
        return (int(cx - w/2), int(cy - h/2), int(w), int(h))

    def get_bbox(self) -> Tuple[int, int, int, int]:
        return self._to_bbox()


class SORTTracker:
    """
    SORT-style multi-object tracker
    ใช้ Kalman filter predict ตำแหน่งแต่ละ track แล้ว Hungarian match กับ detection
    handle crossing ได้ดีเพราะใช้ velocity prediction
    """
    def __init__(self, max_age: int = 30, min_hits: int = 1, iou_thresh: float = 0.15):
        self.max_age   = max_age
        self.min_hits  = min_hits
        self.iou_thresh = iou_thresh
        self.tracks: List[KalmanTrack] = []

    @staticmethod
    def _iou_matrix(preds: List[Tuple], dets: List[Tuple]) -> np.ndarray:
        """คำนวณ IoU matrix ระหว่าง predicted bboxes กับ detections"""
        mat = np.zeros((len(preds), len(dets)))
        for i, p in enumerate(preds):
            for j, d in enumerate(dets):
                px1,py1,px2,py2 = p[0],p[1],p[0]+p[2],p[1]+p[3]
                dx1,dy1,dx2,dy2 = d[0],d[1],d[0]+d[2],d[1]+d[3]
                ix1,iy1 = max(px1,dx1), max(py1,dy1)
                ix2,iy2 = min(px2,dx2), min(py2,dy2)
                inter = max(0,ix2-ix1)*max(0,iy2-iy1)
                if inter == 0:
                    mat[i,j] = 0.0
                    continue
                union = (px2-px1)*(py2-py1)+(dx2-dx1)*(dy2-dy1)-inter
                mat[i,j] = inter/union if union>0 else 0.0
        return mat

    def update(self, detections: List[Tuple[int,int,int,int]]) -> List[Tuple[int, Tuple]]:
        """
        อัปเดต tracker ด้วย detections เฟรมนี้
        Returns: list of (track_id, bbox)
        """
        # predict ทุก track
        predicted = [t.predict() for t in self.tracks]

        matched_t, matched_d = set(), set()
        if self.tracks and detections:
            iou_mat = self._iou_matrix(predicted, detections)
            # Hungarian matching (minimize cost = maximize IoU)
            row_ind, col_ind = linear_sum_assignment(-iou_mat)
            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] >= self.iou_thresh:
                    self.tracks[r].update(detections[c])
                    matched_t.add(r)
                    matched_d.add(c)

        # unmatched detections → new tracks
        for j, det in enumerate(detections):
            if j not in matched_d:
                self.tracks.append(KalmanTrack(det))

        # unmatched tracks → increment no_match
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.no_match += 1

        # ลบ track ที่หายนาน
        self.tracks = [t for t in self.tracks if t.no_match <= self.max_age]

        # return tracks ที่ active
        return [(t.track_id, t.get_bbox())
                for t in self.tracks if t.no_match == 0 and t.hits >= self.min_hits]


# ---------------------------------------------------------------------------
# PersonTracker — ตรวจจับและติดตามคนในวิดีโอ
# ---------------------------------------------------------------------------

class PersonTracker:
    """
    ตรวจจับทุกคนในเฟรม แล้ว track คนที่เลือกไว้ข้ามเฟรม

    v4: tracking แบบ robust ด้วย IoU + velocity prediction + EMA smoothing
    วิธีทำงาน:
    1. ใช้ YOLO11n detect ทุกคนในเฟรม
    2. เฟรมแรก: เรียงซ้าย→ขวา เลือก index
    3. เฟรมถัดไป: matching ด้วย combined score (IoU 60% + centroid 40%)
    4. ทำนายตำแหน่งด้วย velocity (EMA) เพื่อกันกระโดดข้ามคน
    5. smooth bbox ด้วย EMA ให้ภาพนิ่งขึ้น
    """

    def __init__(self, target_person_index: int = 0, target_cx: Optional[float] = None):
        """
        Args:
            target_person_index: ลำดับของคนที่ต้องการ track (fallback ถ้าไม่มี target_cx)
                0 = ซ้ายสุด, 1 = คนถัดไป, ... หรือ -1 = ขวาสุด
            target_cx: x กลางของคนที่ต้องการ (จาก preview) — ถ้ามีจะ lock คนที่ใกล้สุด
        """
        self.target_index = target_person_index
        self._target_cx   = target_cx      # cx จาก preview → ใช้ lock ใน frame แรก
        self._last_bbox: Optional[Tuple[int, int, int, int]] = None
        self._smooth_bbox: Optional[np.ndarray] = None   # EMA smoothed bbox
        self._velocity: np.ndarray = np.zeros(4)          # dx, dy, dw, dh per frame
        self._lost_frames = 0
        self._max_lost = 20
        self._ema_alpha = 0.5   # weight ของเฟรมใหม่ใน EMA
        self._position_history: list = []  # เก็บ 10 ตำแหน่งล่าสุด
        self._history_max = 10
        self._switch_threshold = 0.25  # ต้อง score สูงกว่านี้ถึงจะ switch target

        # โหลด YOLOv8 nano (โมเดลเล็ก ~6MB ดาวน์โหลดอัตโนมัติครั้งแรก)
        try:
            from ultralytics import YOLO
            self._yolo = YOLO("yolo11n.pt")
            self._yolo.overrides["verbose"] = False
            print("✅ YOLO11 โหลดสำเร็จ")
        except ImportError:
            raise ImportError("กรุณาติดตั้ง: pip install ultralytics")

    def detect_people(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        ตรวจจับทุกคนในเฟรมด้วย YOLOv8

        Returns:
            list of (x, y, w, h) เรียงจากซ้ายไปขวา
        """
        results = self._yolo(frame, classes=[0], verbose=False)  # class 0 = person
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return []

        frame_h, frame_w = frame.shape[:2]
        detections = []
        confidences = []
        for box in boxes:
            conf = float(box.conf[0])
            if conf < 0.35:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bw, bh = x2 - x1, y2 - y1
            # คนขอบจอ → threshold เบากว่า
            near_left_edge  = x1 < frame_w * 0.15
            near_right_edge = (x1 + bw) > frame_w * 0.85
            if near_left_edge or near_right_edge:
                min_w, min_h = frame_w * 0.04, frame_h * 0.06
            else:
                min_w, min_h = frame_w * 0.08, frame_h * 0.10
            if bw < min_w or bh < min_h:
                continue
            detections.append((x1, y1, bw, bh))
            confidences.append(conf)

        # NMS เพิ่มเติม — ตัด box ที่ทับซ้อนกัน (IoU > 0.4) ออก
        if len(detections) > 1:
            detections = self._nms(detections, confidences, overlap_thresh=0.4)

        # เรียงจากซ้ายไปขวา
        detections.sort(key=lambda r: r[0] + r[2] / 2)
        return detections

    @staticmethod
    def _iou(a: Tuple, b: Tuple) -> float:
        """คำนวณ IoU ระหว่าง 2 bbox (x,y,w,h)"""
        ax1, ay1 = a[0], a[1]
        ax2, ay2 = a[0] + a[2], a[1] + a[3]
        bx1, by1 = b[0], b[1]
        bx2, by2 = b[0] + b[2], b[1] + b[3]

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0

    def _predicted_bbox(self) -> Optional[Tuple]:
        """คาดเดา bbox ถัดไปจาก velocity"""
        if self._smooth_bbox is None:
            return self._last_bbox
        pred = self._smooth_bbox + self._velocity
        return tuple(pred.astype(int))

    def get_target_bbox(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        คืน bounding box ของคนที่เลือกไว้ในเฟรมนี้ (smooth)

        Returns:
            (x, y, w, h) หรือ None ถ้าหาไม่เจอ
        """
        detections = self.detect_people(frame)

        if len(detections) == 0:
            self._lost_frames += 1
            if self._lost_frames > self._max_lost:
                self._last_bbox = None
                self._smooth_bbox = None
            # คืน smooth bbox ที่ขยับตาม velocity เพื่อกัน freeze
            if self._smooth_bbox is not None:
                self._smooth_bbox = self._smooth_bbox + self._velocity * 0.5
                return tuple(self._smooth_bbox.astype(int))
            return self._last_bbox

        self._lost_frames = 0

        if self._last_bbox is None:
            # เฟรมแรก — เลือกตาม cx จาก preview (ถ้ามี) ไม่งั้นใช้ index
            if self._target_cx is not None and len(detections) > 0:
                # lock คนที่ cx ใกล้ preview มากสุด
                idx = min(range(len(detections)),
                          key=lambda i: abs(detections[i][0] + detections[i][2] / 2 - self._target_cx))
            else:
                idx = max(0, min(self.target_index, len(detections) - 1))
                if self.target_index == -1:
                    idx = len(detections) - 1
            self._last_bbox = detections[idx]
            self._smooth_bbox = np.array(self._last_bbox, dtype=float)
            self._velocity = np.zeros(4)
            self._position_history = [list(self._smooth_bbox)]
            return self._last_bbox

        # ----- matching ด้วย combined score: IoU + centroid + size -----
        predicted = self._predicted_bbox()
        pred_arr = np.array(predicted, dtype=float)
        pred_cx = pred_arr[0] + pred_arr[2] / 2
        pred_cy = pred_arr[1] + pred_arr[3] / 2
        ref_size = max(pred_arr[2], pred_arr[3])

        # ใช้ average velocity จาก history เพื่อ predict ตำแหน่งที่ควรจะเป็น
        if len(self._position_history) >= 3:
            recent = np.array(self._position_history[-3:], dtype=float)
            avg_vel = np.mean(np.diff(recent, axis=0), axis=0)
            extrapolated = np.array(pred_arr) + avg_vel
            ext_cx = extrapolated[0] + extrapolated[2] / 2
            ext_cy = extrapolated[1] + extrapolated[3] / 2
        else:
            ext_cx, ext_cy = pred_cx, pred_cy

        best_score = -float('inf')
        best_bbox  = None

        for bbox in detections:
            iou = self._iou(predicted, bbox)

            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2

            # distance จาก predicted
            dist = ((cx - pred_cx)**2 + (cy - pred_cy)**2)**0.5
            norm_dist = dist / (ref_size + 1e-6)

            # distance จาก extrapolated (velocity-based)
            ext_dist = ((cx - ext_cx)**2 + (cy - ext_cy)**2)**0.5
            norm_ext_dist = ext_dist / (ref_size + 1e-6)

            # size similarity
            size_ratio = min(bbox[2], pred_arr[2]) / (max(bbox[2], pred_arr[2]) + 1e-6)
            size_ratio *= min(bbox[3], pred_arr[3]) / (max(bbox[3], pred_arr[3]) + 1e-6)

            # combined score
            score = (iou * 0.4)                   + (1.0 / (1.0 + norm_dist * 2)) * 0.3                   + (1.0 / (1.0 + norm_ext_dist * 2)) * 0.2                   + size_ratio * 0.1

            if score > best_score:
                best_score = score
                best_bbox  = bbox

        # ถ้า score ต่ำกว่า threshold → อย่า switch, ใช้ค่าเดิม
        if best_score < self._switch_threshold:
            if self._smooth_bbox is not None:
                self._smooth_bbox = self._smooth_bbox + self._velocity * 0.3
                return tuple(self._smooth_bbox.astype(int))
            return self._last_bbox

        # อัปเดต velocity (EMA)
        new_arr = np.array(best_bbox, dtype=float)
        new_vel  = new_arr - self._smooth_bbox if self._smooth_bbox is not None else np.zeros(4)
        self._velocity = 0.7 * self._velocity + 0.3 * new_vel

        # EMA smooth bbox
        if self._smooth_bbox is None:
            self._smooth_bbox = new_arr.copy()
        else:
            self._smooth_bbox = self._ema_alpha * new_arr + (1 - self._ema_alpha) * self._smooth_bbox

        self._last_bbox = best_bbox

        # เก็บ position history
        self._position_history.append(list(new_arr))
        if len(self._position_history) > self._history_max:
            self._position_history.pop(0)

        return tuple(self._smooth_bbox.astype(int))

    @staticmethod
    def _nms(rects, weights, overlap_thresh: float = 0.45):
        """Non-Maximum Suppression (ใช้สำรอง ปัจจุบัน YOLO จัดการให้แล้ว)"""
        if len(rects) == 0:
            return []

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects], dtype=float)
        scores = np.array(weights).flatten()
        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break

            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w_inter = np.maximum(0, xx2 - xx1)
            h_inter = np.maximum(0, yy2 - yy1)
            inter_area = w_inter * h_inter

            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_rest = (boxes[order[1:], 2] - boxes[order[1:], 0]) * \
                        (boxes[order[1:], 3] - boxes[order[1:], 1])

            iou = inter_area / (area_i + area_rest - inter_area + 1e-6)
            order = order[1:][iou < overlap_thresh]

        return [rects[i] for i in keep]


# ---------------------------------------------------------------------------
# Preview — ดูก่อนเลือกคน
# ---------------------------------------------------------------------------

def preview_person_selection(
    video_path: str,
    save_path: str = "person_preview.png",
) -> int:
    """
    แสดงเฟรมแรกสุดที่เจอคนครบ พร้อมหมายเลขกำกับ
    → preview ตรงกับเฟรมที่ tracker จะเริ่ม lock-on จริงๆ

    Returns:
        person_index ที่เลือก (0 เสมอ — user ต้องเลือกเอง)
    """
    tracker = PersonTracker(target_person_index=0)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"ไม่สามารถเปิดวิดีโอ: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  ขนาดวิดีโอ: {raw_w}x{raw_h} | {video_fps:.0f} FPS")

    # ── ขั้น 1: หาจำนวนคนสูงสุดโดย scan 30 เฟรมแรก ───────────────
    print("กำลังนับจำนวนคนในเฟรมแรกๆ...")
    max_people = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(min(60, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        dets = tracker.detect_people(frame)
        max_people = max(max_people, len(dets))

    print(f"  จำนวนคนสูงสุดที่เจอ: {max_people} คน")

    # ── ขั้น 2: หาเฟรมแรกสุดที่เจอคนครบ (≥ max_people) ────────────
    print("กำลังหาเฟรมแรกที่เห็นทุกคนพร้อมกัน...")
    best_frame      = None
    best_detections = []
    best_frame_pos  = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for frame_num in range(min(300, total_frames)):   # สูงสุด 10 วิ (ที่ 30fps)
        ret, frame = cap.read()
        if not ret:
            break
        dets = tracker.detect_people(frame)
        # เลือกเฟรมแรกที่เจอคนครบหรือขาดแค่ 1
        if len(dets) >= max(max_people - 1, 1):
            if best_frame is None or len(dets) > len(best_detections):
                best_frame      = frame.copy()
                best_detections = dets
                best_frame_pos  = frame_num
            # ได้คนครบแล้ว → หยุดทันที ไม่ต้องหาต่อ
            if len(dets) >= max_people:
                break

    cap.release()

    if best_frame is None:
        raise ValueError("ไม่พบคนในวิดีโอ — ตรวจสอบว่าวิดีโอเห็นคนชัดเจน")

    print(f"  ✓ ใช้เฟรม #{best_frame_pos} (t={best_frame_pos/video_fps:.2f}s) — พบ {len(best_detections)} คน")
    print(f"  ⚠️  นี่คือเฟรมเดียวกับที่ tracker จะเริ่ม tracking จริง")

    # ── วาดกรอบ + หมายเลข ──────────────────────────────────────────
    frame_rgb = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
    display   = frame_rgb.copy()

    for i, (x, y, w, h) in enumerate(best_detections):
        color   = PERSON_COLORS[i % len(PERSON_COLORS)]
        r, g, b = color[2], color[1], color[0]
        cv2.rectangle(display, (x, y), (x + w, y + h), (r, g, b), 3)
        lx, ly = max(0, x), max(40, y)
        cv2.rectangle(display, (lx, ly - 40), (lx + 70, ly), (r, g, b), -1)
        cv2.putText(display, f"#{i}", (lx + 6, ly - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    h_img, w_img = display.shape[:2]
    scale = 10.0 / max(w_img, h_img)
    fig, ax = plt.subplots(figsize=(w_img * scale, h_img * scale))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.imshow(display)
    ax.set_title(
        f'เฟรม #{best_frame_pos} (t={best_frame_pos/video_fps:.2f}s) | พบ {len(best_detections)} คน\n'
        f'ดูหมายเลข → ใส่ PERSON_INDEX ใน Cell 3',
        color='white', fontsize=11, pad=10,
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
    plt.show()

    # ── บันทึก cx + frame_pos → ใช้ใน extract_pose_from_video ──────
    preview_data = {
        "frame_pos": int(best_frame_pos),       # ← เฟรมเดียวกับที่ tracker init
        "persons": [
            {"cx": float(x + w / 2), "cy": float(y + h / 2),
             "w": float(w), "h": float(h)}
            for (x, y, w, h) in best_detections
        ],
    }
    data_path = os.path.join(os.path.dirname(save_path), "person_preview_data.json")
    with open(data_path, "w") as f:
        json.dump(preview_data, f)

    print(f"\n→ ใส่ PERSON_INDEX = <หมายเลขที่เห็นในรูป> ใน Cell 3 แล้วรัน Cell 5")
    return 0


def create_person_clip(
    video_path: str,
    person_index: int,
    output_path: str = "person_clip.mp4",
    preview_data_path: Optional[str] = None,
) -> str:
    """
    สร้างวิดีโอที่ crop เฉพาะคนที่เลือก ตลอดทั้งคลิป

    วิธี: YOLO detect ทุกเฟรม (บน frame ที่ resize เล็กลงเพื่อความเร็ว)
    เลือกคนที่ cx ใกล้ smooth_cx เดิมมากที่สุด — ไม่ track by index ตายตัว
    ทำให้ไม่สลับไปที่ว่างและรับมือ crossing ได้ดีกว่า
    """
    from ultralytics import YOLO

    yolo = YOLO("yolo11n.pt")
    yolo.overrides["verbose"] = False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"ไม่สามารถเปิดวิดีโอ: {video_path}")

    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_size = 512
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(output_path, fourcc, video_fps, (out_size, out_size))

    # resize เฟรมให้เล็กลงก่อนส่ง YOLO (เร็วขึ้นมาก แต่ detect ยังดีอยู่)
    detect_w = 960
    scale    = detect_w / orig_w
    detect_h = int(orig_h * scale)

    def yolo_detect_scaled(frame):
        """detect บน frame ที่ resize แล้ว scale bbox กลับขนาดจริง"""
        small = cv2.resize(frame, (detect_w, detect_h))
        res   = yolo(small, classes=[0], verbose=False)[0].boxes
        dets  = []
        if res is None:
            return dets
        for box in res:
            if float(box.conf[0]) < 0.35:
                continue
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            # scale กลับ
            x1,y1,x2,y2 = x1/scale, y1/scale, x2/scale, y2/scale
            bw, bh = x2-x1, y2-y1
            if bw < orig_w*0.04 or bh < orig_h*0.07:
                continue
            dets.append((x1, y1, bw, bh))   # float OK
        return dets

    # แปลง person_index เป็น int เสมอ (กัน string "1")
    person_index = int(person_index)

    print(f"กำลังสร้าง clip สำหรับคน #{person_index}...")

    # --- ลองอ่าน init data จาก preview (ถ้ามี) ---
    if preview_data_path is None:
        # fallback: ลองหาในหลาย path
        for candidate in [
            os.path.join(os.path.dirname(output_path), "person_preview_data.json"),
            "results/person_preview_data.json",
            "person_preview_data.json",
        ]:
            if os.path.exists(candidate):
                preview_data_path = candidate
                break

    smooth_cx = smooth_cy = smooth_half = None

    if preview_data_path:
        with open(preview_data_path) as f:
            pdata = json.load(f)
        persons = pdata["persons"]
        idx = max(0, min(person_index, len(persons)-1))
        p = persons[idx]
        smooth_cx   = p["cx"]
        smooth_cy   = p["cy"]
        smooth_half = max(p["w"], p["h"]) / 2.0 + 60
        print(f"  → อ่านจาก preview: คนที่ {idx} cx={smooth_cx:.0f} cy={smooth_cy:.0f}")
    else:
        # ไม่มี preview data → scan หาเฟรมที่ดี
        print("  ไม่พบ preview data → scan หาเฟรม init...")
        best_init_dets = []
        scan_step = max(1, total_frames // 100)
        for fi in range(0, min(total_frames, int(video_fps * 30)), scan_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, f = cap.read()
            if not ok:
                continue
            d = yolo_detect_scaled(f)
            if len(d) > len(best_init_dets):
                best_init_dets = d
            if len(d) >= 5:
                break
        if not best_init_dets:
            cap.release(); writer.release()
            raise RuntimeError("ไม่พบคนในวิดีโอ")
        best_init_dets.sort(key=lambda d: d[0] + d[2]/2)
        init_idx = max(0, min(person_index, len(best_init_dets)-1))
        det = best_init_dets[init_idx]
        smooth_cx   = det[0] + det[2] / 2.0
        smooth_cy   = det[1] + det[3] / 2.0
        smooth_half = max(det[2], det[3]) / 2.0 + 60
        print(f"  → scan: คนที่ {init_idx} cx={smooth_cx:.0f}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- SORT tracker สำหรับทุกคนพร้อมกัน ---
    KalmanTrack._id_counter = 0
    sort = SORTTracker(max_age=90, min_hits=1, iou_thresh=0.10)
    target_id   = None
    crop_alpha  = 0.25

    # smooth crop window (init จาก preview data)
    smooth_cx   = float(smooth_cx)   if smooth_cx   is not None else None
    smooth_cy   = float(smooth_cy)   if smooth_cy   is not None else None
    smooth_half = float(smooth_half) if smooth_half is not None else None

    pbar = tqdm(total=total_frames, desc="Creating clip")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        dets = yolo_detect_scaled(frame)

        # update SORT ทุกเฟรม (ส่ง detections จริง ไม่ส่งว่าง)
        active = sort.update(dets)  # list of (track_id, bbox)

        # เฟรมแรกที่ได้ active tracks → กำหนด target_id จาก cx ที่ใกล้ preview มากสุด
        if target_id is None and active and smooth_cx is not None:
            best_t = min(active, key=lambda t: abs((t[1][0]+t[1][2]/2) - smooth_cx))
            target_id = best_t[0]
            print(f"  → lock-on track_id={target_id}")

        # หา bbox ของ target track ใน active list
        target_bbox = None
        for tid, bbox in active:
            if tid == target_id:
                target_bbox = bbox
                break

        # ถ้า target หลุดจาก active → ดึง predicted bbox จาก Kalman โดยตรง
        if target_bbox is None and target_id is not None:
            for t in sort.tracks:
                if t.track_id == target_id:
                    target_bbox = t.get_bbox()
                    break

        # smooth crop window
        if target_bbox is not None:
            x, y, w, h = target_bbox
            raw_cx   = x + w / 2.0
            raw_cy   = y + h / 2.0
            raw_half = max(w, h) / 2.0 + 60

            if smooth_cx is None:
                smooth_cx, smooth_cy, smooth_half = raw_cx, raw_cy, raw_half
            else:
                smooth_cx   = crop_alpha*raw_cx   + (1-crop_alpha)*smooth_cx
                smooth_cy   = crop_alpha*raw_cy   + (1-crop_alpha)*smooth_cy
                smooth_half = crop_alpha*raw_half + (1-crop_alpha)*smooth_half

        # crop ตาม smooth window
        if smooth_cx is not None:
            half = int(smooth_half)
            x1 = max(0, int(smooth_cx) - half)
            y1 = max(0, int(smooth_cy) - half)
            x2 = min(orig_w, int(smooth_cx) + half)
            y2 = min(orig_h, int(smooth_cy) + half)
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (out_size,out_size)) if crop.size > 0 else cv2.resize(frame,(out_size,out_size))
        else:
            crop = cv2.resize(frame, (out_size, out_size))

        cv2.putText(crop, f"#{person_index}", (10, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 2, cv2.LINE_AA)
        writer.write(crop)
        pbar.update(1)

    cap.release()
    writer.release()
    pbar.close()
    print(f"✅ บันทึกเสร็จ: {output_path}")
    return output_path

def create_labeled_video(
    video_path: str,
    output_path: str = "labeled_video.mp4",
    output_width: int = 1280,
    preview_seconds: int = 60,
    detect_every: int = 3,
) -> str:
    """
    สร้างวิดีโอที่มีกรอบสี่เหลี่ยมและหมายเลขครอบทุกคน
    ประมวลแค่ preview_seconds วินาทีแรก และ detect ทุก detect_every เฟรม
    เพื่อความเร็ว (~2-3 นาทีแทน 10+ นาที)
    """
    from ultralytics import YOLO
    yolo = YOLO("yolo11n.pt")
    yolo.overrides["verbose"] = False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"ไม่สามารถเปิดวิดีโอ: {video_path}")

    orig_fps     = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    max_frames = min(total_frames, int(orig_fps * preview_seconds))

    scale  = output_width / orig_w
    out_h  = int(orig_h * scale)
    out_w  = output_width

    # output ที่ 30fps เสมอ (ลดขนาดไฟล์ครึ่งนึง)
    out_fps   = min(30.0, orig_fps)
    fps_step  = orig_fps / out_fps   # กี่เฟรม input ต่อ 1 เฟรม output
    detect_every = max(detect_every, int(fps_step))

    # detect บน frame 640px (เร็วที่สุด)
    detect_w  = 640
    det_scale = detect_w / orig_w
    det_h     = int(orig_h * det_scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h))

    colors_bgr = [
        (0,  200, 255),  # #0 เหลือง
        (0,  100, 255),  # #1 ส้ม
        (255, 80,  0 ),  # #2 น้ำเงิน
        (50,  220, 50),  # #3 เขียว
        (255, 0,  180),  # #4 ม่วง
        (0,  220, 220),  # #5 ฟ้า
    ]

    print(f"กำลังสร้าง labeled video (แค่ {preview_seconds} วินาทีแรก)...")
    pbar = tqdm(total=max_frames, desc="Labeling")

    last_dets  = []
    frame_idx  = 0
    next_write = 0.0   # เฟรม input ถัดไปที่จะ write

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # skip เฟรมที่ไม่ต้อง output (ลด fps จาก 60→30)
        if frame_idx < next_write:
            frame_idx += 1
            continue
        next_write += fps_step

        # run YOLO ทุก detect_every เฟรม
        if frame_idx % detect_every == 0:
            small = cv2.resize(frame, (detect_w, det_h))
            res   = yolo(small, classes=[0], verbose=False)[0].boxes
            dets  = []
            if res is not None:
                for box in res:
                    if float(box.conf[0]) < 0.35:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = x1/det_scale, y1/det_scale, x2/det_scale, y2/det_scale
                    bw, bh = x2-x1, y2-y1
                    if bw < orig_w*0.04 or bh < orig_h*0.07:
                        continue
                    dets.append((x1, y1, x2, y2))
                dets.sort(key=lambda d: (d[0]+d[2])/2)
            last_dets = dets

        out_frame = cv2.resize(frame, (out_w, out_h))
        for i, (x1, y1, x2, y2) in enumerate(last_dets):
            color = colors_bgr[i % len(colors_bgr)]
            ox1, oy1 = int(x1*scale), int(y1*scale)
            ox2, oy2 = int(x2*scale), int(y2*scale)
            cv2.rectangle(out_frame, (ox1, oy1), (ox2, oy2), color, 3)
            lx, ly = max(0, ox1), max(50, oy1)
            cv2.rectangle(out_frame, (lx, ly-50), (lx+70, ly), color, -1)
            cv2.putText(out_frame, f"#{i}", (lx+6, ly-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 2, cv2.LINE_AA)

        writer.write(out_frame)
        pbar.update(1)
        frame_idx += 1

    cap.release()
    writer.release()
    pbar.close()
    print(f"✅ บันทึกเสร็จ: {output_path}  ({preview_seconds}s)")
    return output_path


def download_youtube_video(
    url: str,
    output_dir: str = "videos",
    start_time: str = None,
    end_time: str = None,
) -> str:
    """
    ดาวน์โหลดวิดีโอจาก YouTube

    Args:
        url: YouTube URL เช่น "https://www.youtube.com/watch?v=xxxxx"
        output_dir: โฟลเดอร์สำหรับบันทึกวิดีโอ
        start_time: เวลาเริ่ม (format: "MM:SS") เช่น "0:30"
        end_time: เวลาสิ้นสุด เช่น "1:30"

    Returns:
        path ของไฟล์วิดีโอที่ดาวน์โหลดมา
    """
    # ตัด playlist/radio parameters ออก เหลือแค่ video ID
    if "youtube.com" in url or "youtu.be" in url:
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
        parsed = urlparse(url)
        if "youtu.be" in url:
            video_id = parsed.path.lstrip("/").split("?")[0]
        else:
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        if video_id:
            url = f"https://www.youtube.com/watch?v={video_id}"
            print(f"  URL (clean): {url}")

    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-S", "width,res,fps",   # เรียง format ตามความกว้างก่อน → ได้ landscape เสมอ
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        "--remote-components", "ejs:github",   # แก้ signature solving ให้ได้ format ครบ
        "-o", output_template,
        "--no-playlist",
    ]

    if start_time or end_time:
        start = start_time if start_time else "0"
        sections = f"*{start}-{end_time}" if end_time else f"*{start}-inf"
        cmd += ["--download-sections", sections]

    cmd.append(url)

    print(f"กำลังดาวน์โหลดวิดีโอจาก: {url}")
    if start_time or end_time:
        print(f"ช่วงเวลา: {start_time or '0'} → {end_time or 'จบ'}")

    # ให้ subprocess เห็น deno และ tools อื่นๆ เหมือนกับ terminal
    import shutil
    env = os.environ.copy()
    extra_paths = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]
    env["PATH"] = ":".join(extra_paths) + ":" + env.get("PATH", "")

    result = subprocess.run(cmd, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"ดาวน์โหลดไม่สำเร็จ (return code {result.returncode})")

    files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".mp4")]
    if not files:
        raise FileNotFoundError("ไม่พบไฟล์วิดีโอหลังดาวน์โหลด")

    latest_file = max(files, key=os.path.getctime)
    print(f"ดาวน์โหลดสำเร็จ: {latest_file}")
    return latest_file


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Normalize keypoints ให้อยู่ใน scale เดียวกัน"""
    left_hip_idx = DANCE_KEYPOINTS.index(23)
    right_hip_idx = DANCE_KEYPOINTS.index(24)
    left_shoulder_idx = DANCE_KEYPOINTS.index(11)
    right_shoulder_idx = DANCE_KEYPOINTS.index(12)

    hip_center = (keypoints[left_hip_idx] + keypoints[right_hip_idx]) / 2
    shoulder_width = np.linalg.norm(
        keypoints[left_shoulder_idx] - keypoints[right_shoulder_idx]
    )

    if shoulder_width < 1e-6:
        return keypoints - hip_center

    return (keypoints - hip_center) / shoulder_width


def extract_pose_from_video(
    video_path: str,
    sample_fps: float = 10.0,
    person_index: Optional[int] = None,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Extract pose keypoints จากวิดีโอ
    รองรับวิดีโอที่มีหลายคน โดยเลือก track ได้

    Args:
        video_path: path ของวิดีโอ
        sample_fps: จำนวน frame ต่อวินาทีที่จะ extract
        person_index: ลำดับของคนที่ต้องการ track (นับจาก 0 = ซ้ายสุด)
            - None หรือ ไม่ระบุ = ให้ MediaPipe เลือกเอง (เหมาะกับวิดีโอคนเดียว)
            - 0 = คนซ้ายสุด
            - 1 = คนที่สองจากซ้าย
            - -1 = คนขวาสุด
        min_detection_confidence: ค่า confidence ขั้นต่ำ
        min_tracking_confidence: ค่า confidence ขั้นต่ำสำหรับ tracking
        verbose: แสดง progress bar

    Returns:
        dict ที่มี 'keypoints', 'timestamps', 'fps', 'video_info'
    """
    use_tracker = person_index is not None

    # โหลด preview data (cx + frame_pos) เพื่อให้ tracker init ที่เฟรมเดียวกับ preview
    target_cx        = None
    preview_frame_pos = 0   # เฟรมที่ preview ใช้ → tracker จะเริ่ม init ที่นี่
    if use_tracker:
        preview_data_candidates = [
            os.path.join(os.path.dirname(video_path), "person_preview_data.json"),
            "results/person_preview_data.json",
            "person_preview_data.json",
        ]
        for p in preview_data_candidates:
            if os.path.exists(p):
                try:
                    import json as _json
                    pd_data = _json.load(open(p))
                    persons = pd_data.get("persons", [])
                    preview_frame_pos = int(pd_data.get("frame_pos", 0))
                    if person_index < len(persons):
                        target_cx = float(persons[person_index]["cx"])
                        print(f"  → preview data: เฟรม #{preview_frame_pos}, lock-on cx={target_cx:.0f}px (คน #{person_index})")
                except Exception:
                    pass
                break
        if target_cx is None:
            print(f"  → ไม่พบ preview data — ใช้ index={person_index} ในเฟรมแรก")

    tracker = PersonTracker(target_person_index=person_index, target_cx=target_cx) if use_tracker else None

    # ── Init tracker ที่เฟรมเดียวกับ preview ──────────────────────────
    # วิ่งผ่านเฟรมก่อน preview_frame_pos โดยไม่บันทึก เพื่อให้ tracker lock-on
    # ที่ตำแหน่งเดียวกับที่แสดงในภาพ preview
    _init_cap = cv2.VideoCapture(video_path)
    if use_tracker and preview_frame_pos > 0 and _init_cap.isOpened():
        _init_cap.set(cv2.CAP_PROP_POS_FRAMES, preview_frame_pos)
        ret_init, frame_init = _init_cap.read()
        if ret_init:
            tracker.get_target_bbox(frame_init)   # init bbox จากเฟรมนี้
            print(f"  → tracker init ที่เฟรม #{preview_frame_pos} แล้ว")
    _init_cap.release()

    # โหลด pose model (Tasks API — รองรับ mediapipe 0.10+)
    model_path = _get_pose_model(os.path.dirname(os.path.abspath(__file__)))
    base_opts = _mp_python.BaseOptions(model_asset_path=model_path)
    pose_opts = _mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=_mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=min_tracking_confidence,
    )
    landmarker = _mp_vision.PoseLandmarker.create_from_options(pose_opts)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"ไม่สามารถเปิดวิดีโอ: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ตรวจ rotation metadata
    duration = total_frames / video_fps if video_fps > 0 else 0
    frame_interval = max(1, round(video_fps / sample_fps))

    if verbose:
        print(f"\nวิดีโอ: {os.path.basename(video_path)}")
        print(f"  Resolution: {width}x{height} | FPS: {video_fps:.1f} | Duration: {duration:.1f}s")
        if use_tracker:
            pos_name = {0: "ซ้ายสุด", -1: "ขวาสุด"}.get(
                person_index, f"คนที่ {person_index + 1} จากซ้าย"
            )
            print(f"  โหมด: Multi-person → track {pos_name} (index={person_index})")
        else:
            print(f"  โหมด: Single-person (MediaPipe เลือกเอง)")

    keypoints_seq, timestamps, visibility_seq, raw_keypoints_seq = [], [], [], []
    frame_count = 0

    pbar = tqdm(total=total_frames // frame_interval, desc="Extracting poses") if verbose else None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp = frame_count / video_fps

            if use_tracker:
                bbox = tracker.get_target_bbox(frame)
                if bbox is not None:
                    x, y, w, h = bbox
                    pad = int(max(w, h) * 0.15)
                    x1, y1 = max(0, x - pad), max(0, y - pad)
                    x2, y2 = min(width, x + w + pad), min(height, y + h + pad)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        crop = frame
                else:
                    crop = frame
                frame_input = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            else:
                frame_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Tasks API
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_input)
            result = landmarker.detect(mp_img)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                kps = np.array([[lm[i].x, lm[i].y] for i in DANCE_KEYPOINTS], dtype=np.float32)
                vis = np.array([getattr(lm[i], 'visibility', 1.0) for i in DANCE_KEYPOINTS], dtype=np.float32)
                kps_norm = normalize_keypoints(kps)

                # แปลง raw keypoints กลับเป็น full-frame 0-1 coordinates
                if use_tracker and bbox is not None:
                    x, y, w, h = bbox
                    pad = int(max(w, h) * 0.15)
                    x1 = max(0, x - pad); y1 = max(0, y - pad)
                    x2 = min(width, x + w + pad); y2 = min(height, y + h + pad)
                    crop_w, crop_h = x2 - x1, y2 - y1
                    kps_raw = np.stack([
                        (x1 + kps[:, 0] * crop_w) / width,
                        (y1 + kps[:, 1] * crop_h) / height,
                    ], axis=1).astype(np.float32)
                else:
                    kps_raw = kps.copy()  # already full-frame

                keypoints_seq.append(kps_norm)
                raw_keypoints_seq.append(kps_raw)
                visibility_seq.append(vis)
                timestamps.append(timestamp)
            else:
                keypoints_seq.append(None)
                raw_keypoints_seq.append(None)
                visibility_seq.append(None)
                timestamps.append(timestamp)

            if pbar:
                pbar.update(1)

        frame_count += 1

    cap.release()
    landmarker.close()
    if pbar:
        pbar.close()

    # กรองเฉพาะ frame ที่ detect สำเร็จ
    valid_idx  = [i for i, kp in enumerate(keypoints_seq) if kp is not None]
    valid_kps  = [keypoints_seq[i] for i in valid_idx]
    valid_raw  = [raw_keypoints_seq[i] for i in valid_idx]
    valid_ts   = [timestamps[i] for i in valid_idx]
    valid_vis  = [visibility_seq[i] for i in valid_idx]

    detection_rate = len(valid_idx) / max(len(keypoints_seq), 1) * 100

    if verbose:
        print(f"\n✓ สกัดท่าทางสำเร็จ: {len(valid_kps)} frames ({detection_rate:.1f}% detection rate)")
        if detection_rate < 50:
            print("  ⚠️  Detection rate ต่ำ — ลองตรวจสอบ person_index หรือแสงในวิดีโอ")

    return {
        "keypoints": valid_kps,
        "raw_keypoints": valid_raw,
        "visibility": valid_vis,
        "timestamps": valid_ts,
        "fps": video_fps,
        "sample_fps": sample_fps,
        "person_index": person_index,
        "video_info": {
            "path": video_path,
            "width": width, "height": height,
            "total_frames": total_frames, "duration": duration,
        },
        "keypoint_indices": DANCE_KEYPOINTS,
        "keypoint_names": KEYPOINT_NAMES,
    }


def save_pose_data(pose_data: dict, output_path: str) -> None:
    """บันทึก pose data เป็น .npz file"""
    np.savez_compressed(
        output_path,
        keypoints=np.array(pose_data["keypoints"]),
        visibility=np.array(pose_data["visibility"]),
        timestamps=np.array(pose_data["timestamps"]),
        fps=pose_data["fps"],
        sample_fps=pose_data["sample_fps"],
        keypoint_indices=np.array(pose_data["keypoint_indices"]),
    )
    print(f"✓ บันทึก pose data: {output_path}.npz")


def load_pose_data(npz_path: str) -> dict:
    """โหลด pose data จาก .npz file"""
    if not npz_path.endswith(".npz"):
        npz_path += ".npz"
    data = np.load(npz_path, allow_pickle=True)
    return {
        "keypoints": list(data["keypoints"]),
        "visibility": list(data["visibility"]),
        "timestamps": list(data["timestamps"]),
        "fps": float(data["fps"]),
        "sample_fps": float(data["sample_fps"]),
        "keypoint_indices": list(data["keypoint_indices"]),
        "keypoint_names": KEYPOINT_NAMES,
    }


def visualize_pose_on_frame(
    frame: np.ndarray,
    keypoints: np.ndarray,
    color=(0, 255, 0),
    raw_keypoints: np.ndarray = None,
) -> np.ndarray:
    """
    วาด skeleton บน frame
    ถ้าส่ง raw_keypoints (0-1 full-frame) จะวาดตรงตำแหน่งคนจริงๆ
    """
    h, w = frame.shape[:2]
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
        (1, 7), (2, 8), (7, 8), (7, 9), (8, 10), (9, 11), (10, 12),
    ]

    if raw_keypoints is not None:
        # ใช้ตำแหน่งจริง
        def to_pixel(kp):
            return (max(0, min(w-1, int(kp[0] * w))),
                    max(0, min(h-1, int(kp[1] * h))))
        pts = raw_keypoints
    else:
        # fallback: วาดกึ่งกลาง
        center_x, center_y = w // 2, int(h * 0.6)
        scale = w * 0.15
        def to_pixel(kp):
            return (int(center_x + kp[0] * scale), int(center_y + kp[1] * scale))
        pts = keypoints

    for i, j in connections:
        if i < len(pts) and j < len(pts):
            cv2.line(frame, to_pixel(pts[i]), to_pixel(pts[j]), color, 2, cv2.LINE_AA)
    for kp in pts:
        pt = to_pixel(kp)
        cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame
