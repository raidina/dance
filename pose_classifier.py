"""
pose_classifier.py
──────────────────
โหลดโมเดลที่ train ไว้ใน dance_EDA.ipynb

Classes:
    BinaryClassifier — supervised PASS/FAIL (ต้องการ FAIL video)
                       แม่นยำกว่า ใช้จริงเมื่อมีข้อมูล
    AnomalyDetector  — unsupervised PASS/FAIL (ไม่ต้องการ FAIL data)
                       fallback เมื่อยังไม่มี FAIL video
"""

from __future__ import annotations

import numpy as np
import joblib
import json
import os

_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'results', 'pose_cache')


# ──────────────────────────────────────────────
# Binary Classifier (supervised — ต้องการ FAIL data)
# ──────────────────────────────────────────────

class BinaryClassifier:
    """
    Supervised PASS/FAIL classifier ที่ train จาก PASS + FAIL video จริง
    แม่นยำกว่า AnomalyDetector มาก เพราะเคยเห็นตัวอย่างที่ผิดจริงๆ

    โหลดจาก:
        results/pose_cache/pf_classifier.pkl
        results/pose_cache/pf_clf_scaler.pkl
    """

    def __init__(self, model_dir: str = _MODEL_DIR, sample_fps: float = 10.0):
        self.model_dir  = model_dir
        self.sample_fps = sample_fps
        self.model      = None
        self.scaler     = None
        self.is_loaded  = False
        self._load()

    def _load(self):
        clf_path    = os.path.join(self.model_dir, 'pf_classifier.pkl')
        scaler_path = os.path.join(self.model_dir, 'pf_clf_scaler.pkl')
        if not os.path.exists(clf_path) or not os.path.exists(scaler_path):
            return
        try:
            self.model  = joblib.load(clf_path)
            self.scaler = joblib.load(scaler_path)
            self.is_loaded = True
        except Exception as e:
            print(f'BinaryClassifier load error: {e}')

    def predict_frames(self, keypoints: np.ndarray, threshold: float = 0.5) -> list:
        """
        Classify ทุก frame ว่า PASS(1) หรือ FAIL(0)

        Args:
            keypoints : (n_frames, 13, 2)
            threshold : prob ขั้นต่ำสำหรับ PASS

        Returns:
            list of dict: {frame, time, is_pass, pass_prob, label}
        """
        if not self.is_loaded:
            return []

        feats   = extract_pf_features(keypoints, self.sample_fps)
        feats_s = self.scaler.transform(feats)
        probs   = self.model.predict_proba(feats_s)[:, 1]  # prob of PASS

        frames = []
        for i, p in enumerate(probs):
            is_pass = bool(p >= threshold)
            frames.append({
                'frame':     i,
                'time':      float(i / self.sample_fps),
                'is_pass':   is_pass,
                'pass_prob': float(p),
                'label':     'PASS' if is_pass else 'FAIL',
            })
        return frames


# ──────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────

def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    norm = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8
    cos_a = np.einsum('...i,...i', ba, bc) / norm
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def extract_features(kps: np.ndarray) -> np.ndarray:
    """42 scale-invariant features: positions + angles + distances + tilt"""
    hip_c  = (kps[:, 7] + kps[:, 8]) / 2
    shld_w = np.linalg.norm(kps[:, 1] - kps[:, 2], axis=1, keepdims=True) + 1e-8
    kn     = (kps - hip_c[:, None, :]) / shld_w[:, None, :]

    feat_pos = kn.reshape(len(kps), -1)   # 26

    angles = np.column_stack([
        compute_angle(kps[:, 1],  kps[:, 3],  kps[:, 5]),
        compute_angle(kps[:, 2],  kps[:, 4],  kps[:, 6]),
        compute_angle(kps[:, 7],  kps[:, 9],  kps[:, 11]),
        compute_angle(kps[:, 8],  kps[:, 10], kps[:, 12]),
        compute_angle(kps[:, 3],  kps[:, 1],  kps[:, 7]),
        compute_angle(kps[:, 4],  kps[:, 2],  kps[:, 8]),
        compute_angle(kps[:, 1],  kps[:, 7],  kps[:, 9]),
        compute_angle(kps[:, 2],  kps[:, 8],  kps[:, 10]),
    ]) / 180.0   # 8

    dists = np.hstack([
        np.linalg.norm(kn[:, 5]  - kn[:, 6],  axis=1, keepdims=True),
        np.linalg.norm(kn[:, 11] - kn[:, 12], axis=1, keepdims=True),
        np.linalg.norm(kn[:, 5]  - kn[:, 7],  axis=1, keepdims=True),
        np.linalg.norm(kn[:, 6]  - kn[:, 8],  axis=1, keepdims=True),
        np.linalg.norm(kn[:, 5]  - kn[:, 0],  axis=1, keepdims=True),
        np.linalg.norm(kn[:, 6]  - kn[:, 0],  axis=1, keepdims=True),
    ])   # 6

    tilt = np.hstack([
        (kn[:, 1, 1] - kn[:, 2, 1]).reshape(-1, 1),
        (kn[:, 7, 1] - kn[:, 8, 1]).reshape(-1, 1),
    ])   # 2

    return np.hstack([feat_pos, angles, dists, tilt]).astype(np.float32)


def compute_speed(kps: np.ndarray, fps: float = 10.0) -> np.ndarray:
    """ความเร็วการเคลื่อนที่ของแต่ละ joint → (n_frames, 13)"""
    if len(kps) < 2:
        return np.zeros((len(kps), kps.shape[1]))
    delta = np.diff(kps, axis=0)
    speed = np.linalg.norm(delta, axis=2) * fps
    return np.vstack([speed[:1], speed])


def compute_acceleration(kps: np.ndarray, fps: float = 10.0) -> np.ndarray:
    """acceleration ของแต่ละ joint (rate of change of speed) → (n_frames, 13)
    จับ impulse/จังหวะของท่าเต้น — ต่างกันชัดเจนระหว่างเพลง"""
    speed = compute_speed(kps, fps)          # (n_frames, 13)
    if len(speed) < 2:
        return np.zeros_like(speed)
    acc = np.diff(speed, axis=0) * fps       # (n_frames-1, 13)
    return np.vstack([acc[:1], acc])         # (n_frames, 13)


def extract_pf_features(kps: np.ndarray, fps: float = 10.0) -> np.ndarray:
    """68 features = 42 pose + 13 speed + 13 acceleration"""
    return np.hstack([
        extract_features(kps),
        compute_speed(kps, fps),
        compute_acceleration(kps, fps),
    ]).astype(np.float32)


# ──────────────────────────────────────────────
# Anomaly Detector
# ──────────────────────────────────────────────

class AnomalyDetector:
    """
    ตรวจจับว่าแต่ละ frame เต้น PASS หรือ FAIL
    โดยใช้ Isolation Forest ที่ train จาก reference (PASS) อย่างเดียว

    ไม่ต้องการ FAIL data — โมเดลเรียนรู้ว่า "ท่าที่ถูกต้องมีลักษณะยังไง"
    แล้วบอกว่า user frame ไหนผิดปกติ
    """

    def __init__(self, model_dir: str = _MODEL_DIR, sample_fps: float = 10.0):
        self.model_dir  = model_dir
        self.sample_fps = sample_fps
        self.detector   = None
        self.scaler     = None
        self.meta       = {}
        self.is_loaded  = False
        self._load()

    def _load(self):
        required = ['anomaly_detector.pkl', 'pf_scaler.pkl']
        for fname in required:
            if not os.path.exists(os.path.join(self.model_dir, fname)):
                return
        try:
            self.detector = joblib.load(os.path.join(self.model_dir, 'anomaly_detector.pkl'))
            self.scaler   = joblib.load(os.path.join(self.model_dir, 'pf_scaler.pkl'))
            meta_path = os.path.join(self.model_dir, 'pf_meta.json')
            if os.path.exists(meta_path):
                self.meta = json.load(open(meta_path))
            self.is_loaded = True
        except Exception as e:
            print(f'AnomalyDetector load error: {e}')

    def predict_frames(
        self, keypoints: np.ndarray, smooth_window: int = 5, threshold: float = 0.5
    ) -> list:
        """
        Classify ทุก frame ว่า PASS หรือ FAIL

        Args:
            keypoints    : (n_frames, 13, 2)
            smooth_window: rolling window สำหรับ smooth (frames)
            threshold    : เกณฑ์ตัดสิน 0-1 (สูง=เข้มงวด)

        Returns:
            list of dict: {frame, time, is_pass, pass_prob, label}
        """
        if not self.is_loaded:
            return [{'frame': i, 'time': i / self.sample_fps,
                     'is_pass': True, 'pass_prob': 1.0, 'label': 'PASS'}
                    for i in range(len(keypoints))]

        feats   = extract_pf_features(keypoints, self.sample_fps)
        feats_s = self.scaler.transform(feats)

        # anomaly score: สูง = ปกติ (PASS), ต่ำ = ผิดปกติ (FAIL)
        scores = self.detector.decision_function(feats_s)

        # normalize เป็น 0-1 โดยใช้ range จาก reference
        s_min = self.meta.get('score_min', scores.min())
        s_max = self.meta.get('score_max', scores.max())
        probs = (scores - s_min) / (s_max - s_min + 1e-8)
        probs = np.clip(probs, 0, 1)

        # rolling mean smoothing
        w            = max(1, smooth_window)
        kernel       = np.ones(w) / w
        probs_smooth = np.convolve(probs, kernel, mode='same')

        fps    = self.sample_fps
        frames = []
        for i, p in enumerate(probs_smooth):
            is_pass = bool(p >= threshold)
            frames.append({
                'frame':     i,
                'time':      float(i / fps),
                'is_pass':   is_pass,
                'pass_prob': float(p),
                'label':     'PASS' if is_pass else 'FAIL',
            })
        return frames
