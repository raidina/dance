"""
pose_guide_generator.py
------------------------
สร้าง PDF คู่มือสอนเต้น จากวิดีโอ K-pop

ขั้นตอน:
1. ตรวจจับ "ท่าหยุดนิ่ง" โดยหาช่วงที่ร่างกายเคลื่อนไหวน้อยที่สุด
2. ดึงภาพจริงจากวิดีโอ ณ จังหวะนั้น
3. วาด skeleton ทับบนภาพจริง
4. รวมทุกท่าออกมาเป็น PDF แบบ Flipbook
"""

import os
import cv2
import numpy as np
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision

def _get_pose_model(model_dir: str = ".") -> str:
    path = os.path.join(model_dir, "pose_landmarker_lite.task")
    if not os.path.exists(path):
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "pose_landmarker/pose_landmarker_lite/float16/1/"
               "pose_landmarker_lite.task")
        print("กำลังดาวน์โหลด pose model (~5MB)...")
        urllib.request.urlretrieve(url, path)
    return path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Image, Table, TableStyle,
    Paragraph, Spacer, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from tqdm import tqdm
from PIL import Image as PILImage
import io
from typing import List, Optional

from reference_processor import (
    PersonTracker,
    DANCE_KEYPOINTS,
    normalize_keypoints,
)


# ---------------------------------------------------------------------------
# Skeleton drawing constants
# ---------------------------------------------------------------------------

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),    # nose → shoulders
    (1, 3), (2, 4),    # shoulders → elbows
    (3, 5), (4, 6),    # elbows → wrists
    (1, 7), (2, 8),    # shoulders → hips
    (7, 8),             # hip ↔ hip
    (7, 9), (8, 10),   # hips → knees
    (9, 11), (10, 12), # knees → ankles
]

# สีของ skeleton แต่ละส่วน (BGR)
SEGMENT_COLORS = {
    "arms":  (255, 80,  80),   # แดง — แขน
    "torso": (80,  200, 80),   # เขียว — ลำตัว
    "legs":  (80,  80,  255),  # น้ำเงิน — ขา
}

CONNECTION_COLORS = [
    SEGMENT_COLORS["torso"],   # nose-left_shoulder
    SEGMENT_COLORS["torso"],   # nose-right_shoulder
    SEGMENT_COLORS["arms"],    # left_shoulder-left_elbow
    SEGMENT_COLORS["arms"],    # right_shoulder-right_elbow
    SEGMENT_COLORS["arms"],    # left_elbow-left_wrist
    SEGMENT_COLORS["arms"],    # right_elbow-right_wrist
    SEGMENT_COLORS["torso"],   # left_shoulder-left_hip
    SEGMENT_COLORS["torso"],   # right_shoulder-right_hip
    SEGMENT_COLORS["torso"],   # left_hip-right_hip
    SEGMENT_COLORS["legs"],    # left_hip-left_knee
    SEGMENT_COLORS["legs"],    # right_hip-right_knee
    SEGMENT_COLORS["legs"],    # left_knee-left_ankle
    SEGMENT_COLORS["legs"],    # right_knee-right_ankle
]


# ---------------------------------------------------------------------------
# Step 1: ตรวจจับท่าหยุดนิ่ง
# ---------------------------------------------------------------------------

def detect_key_poses(
    video_path: str,
    person_index: Optional[int] = None,
    sample_fps: float = 10.0,
    motion_threshold: float = 0.03,
    min_hold_frames: int = 3,
    max_poses: int = 20,
    verbose: bool = True,
    mode: str = "hold",
) -> List[dict]:
    """
    ตรวจจับท่าจากวิดีโอ

    Args:
        video_path: path ของวิดีโอ
        person_index: ลำดับคนที่ต้องการ (None = คนเดียว)
        sample_fps: fps ที่จะ sample
        motion_threshold: threshold movement (0-1)
        min_hold_frames: จำนวน frame ขั้นต่ำที่ต้องหยุดนิ่ง (mode=hold)
        max_poses: จำนวนท่าสูงสุดที่จะดึงออกมา
        verbose: แสดง progress
        mode: 'hold' = เลือกเฟรมที่หยุดนิ่ง (default)
              'motion' = เลือกเฟรมที่มีการขยับสูงสุด

    Returns:
        list of dict: {'frame_idx', 'timestamp', 'keypoints', 'movement'}
    """
    tracker = PersonTracker(target_person_index=person_index) if person_index is not None else None

    # โหลด pose model (Tasks API)
    model_path = _get_pose_model(os.path.dirname(os.path.abspath(__file__)))
    base_opts = _mp_python.BaseOptions(model_asset_path=model_path)
    pose_opts = _mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=_mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = _mp_vision.PoseLandmarker.create_from_options(pose_opts)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"ไม่สามารถเปิดวิดีโอ: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / sample_fps))

    if verbose:
        print(f"\n[Pose Detector] วิดีโอ: {os.path.basename(video_path)}")
        print(f"  FPS: {video_fps:.1f} | Sample: {sample_fps} fps | Threshold: {motion_threshold}")

    all_frames = []
    frame_count = 0
    pbar = tqdm(total=total_frames // frame_interval, desc="Scanning poses") if verbose else None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp = frame_count / video_fps

            if tracker:
                bbox = tracker.get_target_bbox(frame)
                if bbox:
                    x, y, w, h = bbox
                    pad = int(max(w, h) * 0.15)
                    x1, y1 = max(0, x - pad), max(0, y - pad)
                    x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
                    crop = frame[y1:y2, x1:x2]
                    frame_input = cv2.cvtColor(crop if crop.size > 0 else frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_input)
            result = landmarker.detect(mp_img)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                kps = np.array([[lm[i].x, lm[i].y] for i in DANCE_KEYPOINTS], dtype=np.float32)
                kps_norm = normalize_keypoints(kps)
                all_frames.append({
                    "frame_idx": frame_count,
                    "timestamp": timestamp,
                    "keypoints": kps_norm,
                    "raw_keypoints": kps,  # ตำแหน่งจริง 0-1 ของ MediaPipe
                })

            if pbar:
                pbar.update(1)

        frame_count += 1

    cap.release()
    landmarker.close()
    if pbar:
        pbar.close()

    if len(all_frames) < 2:
        raise ValueError("ตรวจจับท่าทางได้น้อยเกินไป กรุณาตรวจสอบวิดีโอ")

    # คำนวณ movement score ระหว่าง frame
    for i in range(1, len(all_frames)):
        kp_prev = all_frames[i - 1]["keypoints"]
        kp_curr = all_frames[i]["keypoints"]
        movement = float(np.mean(np.linalg.norm(kp_curr - kp_prev, axis=1)))
        all_frames[i]["movement"] = movement
    all_frames[0]["movement"] = all_frames[1]["movement"]

    # ============================================================
    # MODE: hold — เลือกเฟรมที่หยุดนิ่ง (movement ต่ำ)
    # ============================================================
    if mode == "hold":
        hold_segments = []
        in_hold = False
        hold_start = 0

        for i, f in enumerate(all_frames):
            if f["movement"] < motion_threshold:
                if not in_hold:
                    in_hold = True
                    hold_start = i
            else:
                if in_hold and (i - hold_start) >= min_hold_frames:
                    hold_segments.append((hold_start, i - 1))
                in_hold = False

        if in_hold and (len(all_frames) - hold_start) >= min_hold_frames:
            hold_segments.append((hold_start, len(all_frames) - 1))

        if verbose:
            print(f"  พบ {len(hold_segments)} ช่วงที่หยุดนิ่ง")

        key_poses = []
        for start, end in hold_segments:
            segment = all_frames[start:end + 1]
            best = min(segment, key=lambda f: f["movement"])
            key_poses.append(best)

        # fallback ถ้าหา hold ไม่เจอพอ
        if len(key_poses) < 5:
            if verbose:
                print(f"  ⚠️ hold segment น้อย ({len(key_poses)}) — เพิ่มด้วย low-movement sampling")
            sorted_frames = sorted(all_frames[1:], key=lambda f: f["movement"])
            added = set(f["frame_idx"] for f in key_poses)
            for f in sorted_frames:
                if f["frame_idx"] not in added:
                    key_poses.append(f)
                    added.add(f["frame_idx"])
                if len(key_poses) >= max_poses:
                    break

    # ============================================================
    # MODE: motion — เลือกเฟรมที่มีการขยับสูงสุด
    # ============================================================
    elif mode == "motion":
        # หาช่วงที่ movement สูงต่อเนื่อง (motion burst segments)
        motion_segments = []
        in_motion = False
        motion_start = 0

        for i, f in enumerate(all_frames):
            if f["movement"] > motion_threshold:
                if not in_motion:
                    in_motion = True
                    motion_start = i
            else:
                if in_motion:
                    motion_segments.append((motion_start, i - 1))
                in_motion = False

        if in_motion:
            motion_segments.append((motion_start, len(all_frames) - 1))

        if verbose:
            print(f"  พบ {len(motion_segments)} ช่วงที่มีการขยับ")

        key_poses = []
        for start, end in motion_segments:
            segment = all_frames[start:end + 1]
            # เลือก frame ที่ movement สูงสุดในแต่ละ burst
            best = max(segment, key=lambda f: f["movement"])
            key_poses.append(best)

        # ถ้าน้อยเกินไป เพิ่มจาก high-movement frames
        if len(key_poses) < 5:
            if verbose:
                print(f"  ⚠️ motion segment น้อย — เพิ่มด้วย high-movement sampling")
            sorted_frames = sorted(all_frames[1:], key=lambda f: f["movement"], reverse=True)
            added = set(f["frame_idx"] for f in key_poses)
            for f in sorted_frames:
                if f["frame_idx"] not in added:
                    key_poses.append(f)
                    added.add(f["frame_idx"])
                if len(key_poses) >= max_poses:
                    break

    # ============================================================
    # MODE: filmstrip — sample ทุก interval_sec วินาที
    # ============================================================
    elif mode == "filmstrip":
        key_poses = []
        if len(all_frames) == 0:
            pass
        else:
            # หา interval เป็นจำนวน all_frames index
            # all_frames ถูก sample ที่ sample_fps แล้ว
            # interval_sec ถูกส่งมาผ่าน motion_threshold ชั่วคราว (reuse param)
            # แต่เราจะใช้ sample_fps เพื่อคำนวณ
            step = max(1, round(sample_fps * 0.5))  # 0.5 วินาที
            for i in range(0, len(all_frames), step):
                key_poses.append(all_frames[i])

    else:
        raise ValueError(f"mode ต้องเป็น 'hold', 'motion', หรือ 'filmstrip' เท่านั้น ได้รับ: {mode!r}")

    # เรียงตาม timestamp และจำกัดจำนวน
    key_poses.sort(key=lambda f: f["timestamp"])
    key_poses = key_poses[:max_poses]

    if verbose:
        mode_label = {"hold": "ท่าหยุดนิ่ง", "motion": "ท่าขยับ", "filmstrip": "ท่าต่อเนื่อง"}.get(mode, "ท่า")
        print(f"  ✓ เลือกได้ {len(key_poses)} {mode_label}")

    return key_poses


# ---------------------------------------------------------------------------
# Step 2: วาด skeleton บนภาพจริง
# ---------------------------------------------------------------------------

def draw_skeleton_on_frame(
    frame: np.ndarray,
    keypoints_normalized: np.ndarray,
    line_thickness: int = 3,
    circle_radius: int = 6,
    alpha: float = 0.85,
    raw_keypoints: np.ndarray = None,
) -> np.ndarray:
    """
    วาด skeleton บนภาพจริงจากวิดีโอ

    Args:
        frame: ภาพ BGR จากวิดีโอ
        keypoints_normalized: normalized keypoints (N, 2) — ใช้ fallback เท่านั้น
        line_thickness: ความหนาของเส้น
        circle_radius: ขนาดจุด joint
        alpha: ความโปร่งใสของ skeleton overlay (0-1)
        raw_keypoints: ตำแหน่งจริง 0-1 จาก MediaPipe (แนะนำให้ส่งมาเสมอ)

    Returns:
        frame ที่มี skeleton วาดทับ
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    if raw_keypoints is not None:
        # ใช้ตำแหน่งจริงจาก MediaPipe (0-1 range) → pixel
        def to_pixel(kp):
            px = int(kp[0] * w)
            py = int(kp[1] * h)
            return (max(0, min(w - 1, px)), max(0, min(h - 1, py)))
        pts = [to_pixel(kp) for kp in raw_keypoints]
    else:
        # fallback: ประมาณจาก normalized keypoints
        hip_cx = w * 0.50
        hip_cy = h * 0.60
        scale  = w * 0.15
        def to_pixel(kp):
            px = int(hip_cx + kp[0] * scale)
            py = int(hip_cy + kp[1] * scale)
            return (max(0, min(w - 1, px)), max(0, min(h - 1, py)))
        pts = [to_pixel(kp) for kp in keypoints_normalized]

    # วาด connections
    for idx, (i, j) in enumerate(SKELETON_CONNECTIONS):
        if i < len(pts) and j < len(pts):
            color = CONNECTION_COLORS[idx % len(CONNECTION_COLORS)]
            cv2.line(overlay, pts[i], pts[j], color, line_thickness, cv2.LINE_AA)

    # วาด joints
    for pt in pts:
        cv2.circle(overlay, pt, circle_radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, pt, circle_radius - 2, (40, 40, 40), -1, cv2.LINE_AA)

    # blend กับ frame เดิม
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return result


def get_frame_at(video_path: str, frame_idx: int) -> np.ndarray:
    """ดึง frame ที่ตำแหน่ง frame_idx จากวิดีโอ"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"ไม่สามารถดึง frame ที่ {frame_idx}")
    return frame


def prepare_pose_images(
    video_path: str,
    key_poses: List[dict],
    target_size: tuple = (480, 640),
) -> List[np.ndarray]:
    """
    ดึงภาพจริงและวาด skeleton สำหรับทุกท่า

    Args:
        video_path: path วิดีโอ
        key_poses: ผลจาก detect_key_poses()
        target_size: (width, height) ของภาพ output

    Returns:
        list ของ BGR images พร้อม skeleton
    """
    images = []
    for pose in tqdm(key_poses, desc="Preparing images"):
        frame = get_frame_at(video_path, pose["frame_idx"])

        # resize ให้พอดี
        tw, th = target_size
        fh, fw = frame.shape[:2]
        scale = min(tw / fw, th / fh)
        new_w, new_h = int(fw * scale), int(fh * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h))

        # pad ให้ได้ขนาดตาม target
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        y_off = (th - new_h) // 2
        x_off = (tw - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = frame_resized

        # วาด skeleton (ใช้ raw_keypoints ถ้ามี เพื่อตำแหน่งที่ถูกต้อง)
        result = draw_skeleton_on_frame(
            canvas,
            pose["keypoints"],
            raw_keypoints=pose.get("raw_keypoints"),
        )
        images.append(result)

    return images


# ---------------------------------------------------------------------------
# Step 3: สร้าง PDF
# ---------------------------------------------------------------------------

def bgr_to_pil(img_bgr: np.ndarray) -> PILImage.Image:
    """แปลง OpenCV BGR → PIL Image"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return PILImage.fromarray(img_rgb)


def create_dance_guide_pdf(
    video_path: str,
    key_poses: List[dict],
    pose_images: List[np.ndarray],
    output_path: str,
    title: str = "K-Pop Dance Guide",
    poses_per_row: int = 2,
    song_name: str = "",
) -> str:
    """
    สร้าง PDF คู่มือท่าเต้น

    Args:
        video_path: path ของวิดีโอ (สำหรับ metadata)
        key_poses: ผลจาก detect_key_poses()
        pose_images: ผลจาก prepare_pose_images()
        output_path: path สำหรับบันทึก PDF
        title: ชื่อ PDF
        poses_per_row: จำนวนท่าต่อแถว (2 = ดูง่ายสุด)
        song_name: ชื่อเพลง/ศิลปิน

    Returns:
        output_path ของ PDF ที่สร้าง
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Page size A4
    PAGE_W, PAGE_H = A4  # 595 x 842 pt
    margin = 1.5 * cm

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#555555"),
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    pose_label_style = ParagraphStyle(
        "PoseLabel",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#333333"),
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    legend_style = ParagraphStyle(
        "Legend",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#666666"),
        alignment=TA_LEFT,
    )

    story = []

    # ---- หน้าปก ----
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph(title, title_style))
    if song_name:
        story.append(Paragraph(song_name, subtitle_style))
    story.append(Paragraph(f"รวม {len(key_poses)} ท่าหลัก", subtitle_style))
    story.append(Spacer(1, 0.5 * cm))

    # Legend (คำอธิบายสี skeleton)
    legend_data = [[
        Paragraph("<font color='#FF5050'>━━</font> แขน", legend_style),
        Paragraph("<font color='#50C850'>━━</font> ลำตัว", legend_style),
        Paragraph("<font color='#5050FF'>━━</font> ขา", legend_style),
        Paragraph("● จุด joint", legend_style),
    ]]
    legend_table = Table(legend_data, colWidths=[3.5 * cm] * 4)
    legend_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f5f5")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROUNDEDCORNERS", [4]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(legend_table)
    story.append(PageBreak())

    # ---- หน้าท่าเต้น ----
    # คำนวณขนาดภาพให้พอดีหน้า
    usable_w = PAGE_W - 2 * margin
    img_w = (usable_w - (poses_per_row - 1) * 0.3 * cm) / poses_per_row
    img_h = img_w * (640 / 480)  # สัดส่วน 4:3

    # จัดเป็น row
    rows = []
    for i in range(0, len(key_poses), poses_per_row):
        chunk = list(zip(key_poses[i:i + poses_per_row], pose_images[i:i + poses_per_row]))
        rows.append(chunk)

    for row_idx, row in enumerate(rows):
        # แถวภาพ
        img_row = []
        for pose, img_bgr in row:
            pil_img = bgr_to_pil(img_bgr)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=90)
            buf.seek(0)
            rl_img = Image(buf, width=img_w, height=img_h)
            img_row.append(rl_img)

        # เติม cell ว่างถ้า row ไม่เต็ม
        while len(img_row) < poses_per_row:
            img_row.append(Paragraph("", pose_label_style))

        # แถว label
        label_row = []
        for k, (pose, _) in enumerate(row):
            pose_num = row_idx * poses_per_row + k + 1
            ts = pose["timestamp"]
            mins = int(ts // 60)
            secs = ts % 60
            time_str = f"{mins}:{secs:05.2f}"
            label_row.append(
                Paragraph(f"<b>ท่าที่ {pose_num}</b><br/>⏱ {time_str}", pose_label_style)
            )

        while len(label_row) < poses_per_row:
            label_row.append(Paragraph("", pose_label_style))

        col_widths = [img_w] * poses_per_row
        tbl = Table([img_row, label_row], colWidths=col_widths)
        tbl.setStyle(TableStyle([
            ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",      (0, 0), (-1, 0),  "MIDDLE"),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING",(0, 0), (-1, -1), 4),
            ("BOX",         (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("LINEBELOW",   (0, 0), (-1, 0),  0.5, colors.HexColor("#eeeeee")),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.4 * cm))

        # ขึ้นหน้าใหม่ทุก 2 แถว (4 ท่า)
        if (row_idx + 1) % 2 == 0 and row_idx < len(rows) - 1:
            story.append(PageBreak())

    doc.build(story)
    print(f"\n✅ สร้าง PDF สำเร็จ: {output_path}")
    print(f"   {len(key_poses)} ท่า | {len(rows)} แถว")
    return output_path


# ---------------------------------------------------------------------------
# All-in-one function
# ---------------------------------------------------------------------------

def generate_dance_guide(
    video_path: str,
    output_path: str = "results/dance_guide.pdf",
    person_index: Optional[int] = None,
    max_poses: int = 32,
    poses_per_row: int = 2,
    song_name: str = "",
    motion_threshold: float = 0.03,
    mode: str = "hold",
) -> str:
    """
    ฟังก์ชันหลัก — รันทุกอย่างในครั้งเดียว

    Args:
        video_path: path วิดีโอ K-pop
        output_path: path output PDF
        person_index: คนที่ต้องการ track (None = คนเดียว)
        max_poses: จำนวนท่าสูงสุด
        poses_per_row: จำนวนท่าต่อแถวใน PDF
        song_name: ชื่อเพลง/ศิลปิน แสดงในหน้าปก
        motion_threshold: ความ sensitive ในการตรวจจับท่า
        mode: 'hold' = ท่าหยุดนิ่ง | 'motion' = ท่าขยับ | 'filmstrip' = ทุก 0.5 วินาที

    Returns:
        path ของ PDF ที่สร้าง
    """
    print("="*50)
    print("  📖 K-Pop Dance Guide Generator")
    print("="*50)

    # 1. ตรวจจับท่าหยุดนิ่ง
    print("\n[1/3] ตรวจจับท่าหลัก...")
    key_poses = detect_key_poses(
        video_path,
        person_index=person_index,
        motion_threshold=motion_threshold,
        max_poses=max_poses,
        mode=mode,
    )

    # 2. เตรียมภาพ
    print("\n[2/3] เตรียมภาพพร้อม skeleton...")
    pose_images = prepare_pose_images(video_path, key_poses)

    # 3. สร้าง PDF
    print("\n[3/3] สร้าง PDF คู่มือ...")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    title = song_name if song_name else f"Dance Guide: {video_name[:40]}"

    return create_dance_guide_pdf(
        video_path=video_path,
        key_poses=key_poses,
        pose_images=pose_images,
        output_path=output_path,
        title=title,
        poses_per_row=poses_per_row,
        song_name=song_name,
    )


if __name__ == "__main__":
    print("=== Dance Guide Generator ===")
    print("ใช้งานผ่าน Notebook หรือ:")
    print('from pose_guide_generator import generate_dance_guide')
    print('generate_dance_guide("video.mp4", "results/guide.pdf", song_name="BLACKPINK - Pink Venom")')
