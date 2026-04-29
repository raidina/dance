"""
dance_comparator.py
--------------------
เปรียบเทียบท่าเต้นของผู้เรียนกับ reference (ศิลปิน K-pop)
คำนวณ similarity score และสร้าง visualization
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from fastdtw import fastdtw
from scipy.spatial.distance import cosine
from scipy.signal import savgol_filter
from tqdm import tqdm
from typing import Optional
import subprocess
import tempfile

from reference_processor import (
    extract_pose_from_video,
    save_pose_data,
    load_pose_data,
    visualize_pose_on_frame,
    DANCE_KEYPOINTS,
    KEYPOINT_NAMES,
)
try:
    from pose_classifier import AnomalyDetector as _AnomalyDetector
    _CLF_AVAILABLE = True
except Exception:
    _AnomalyDetector = None
    _CLF_AVAILABLE = False

# lazy singleton
_pf_clf = None

def _get_pf_classifier(sample_fps: float = 10.0):
    global _pf_clf
    if not _CLF_AVAILABLE: return None
    if _pf_clf is None: _pf_clf = _AnomalyDetector(sample_fps=sample_fps)
    return _pf_clf


# ========== Similarity Functions ==========

def keypoints_to_vector(keypoints: np.ndarray) -> np.ndarray:
    """แปลง keypoints (N, 2) → flat vector (N*2,) สำหรับ DTW"""
    return keypoints.flatten()


def compute_frame_similarity(kp1: np.ndarray, kp2: np.ndarray) -> float:
    """
    คำนวณ similarity ของสอง pose (per frame)
    ใช้ cosine similarity บน flattened keypoints

    Returns:
        similarity score 0.0 - 1.0 (1.0 = เหมือนกันทุกประการ)
    """
    v1 = keypoints_to_vector(kp1)
    v2 = keypoints_to_vector(kp2)

    # Cosine similarity
    sim = 1.0 - cosine(v1, v2)
    return max(0.0, min(1.0, sim))


def align_sequences_dtw(seq1: list, seq2: list) -> tuple:
    """
    จัดเรียง sequence สองชุดด้วย Dynamic Time Warping

    Args:
        seq1: list of keypoints arrays (reference)
        seq2: list of keypoints arrays (user)

    Returns:
        (distance, path) ของ DTW
    """
    vectors1 = [keypoints_to_vector(kp) for kp in seq1]
    vectors2 = [keypoints_to_vector(kp) for kp in seq2]
    distance, path = fastdtw(vectors1, vectors2, dist=cosine)
    return distance, path


def compute_similarity_along_path(
    seq1: list, seq2: list, dtw_path: list
) -> np.ndarray:
    """
    คำนวณ similarity score ตาม DTW path

    Returns:
        array ของ similarity scores ตาม path
    """
    similarities = []
    for i, j in dtw_path:
        sim = compute_frame_similarity(seq1[i], seq2[j])
        similarities.append(sim)
    return np.array(similarities)


def smooth_scores(scores: np.ndarray, window: int = 11) -> np.ndarray:
    """ทำให้ score curve เรียบขึ้นด้วย Savitzky-Golay filter"""
    if len(scores) < window:
        return scores
    return savgol_filter(scores, window_length=min(window, len(scores) - 1 if len(scores) % 2 == 0 else len(scores)), polyorder=2)


def _compute_dtw_penalty(user_distance: float, baseline: float, path_len: int) -> float:
    """
    คำนวณ penalty จาก cosine distance ต่อ frame (absolute threshold)

    cosine distance อยู่ระหว่าง 0-2 เสมอ:
      per_frame < 0.08  → เต้นถูก ใกล้เคียงมาก   → penalty 1.0
      per_frame 0.08-0.15 → เต้นได้ มีผิดบ้าง    → penalty 0.9-0.8
      per_frame 0.15-0.25 → เต้นผิดบ้าง          → penalty 0.7-0.5
      per_frame 0.25-0.40 → เต้นผิดเยอะ           → penalty 0.4-0.2
      per_frame > 0.40    → น่าจะเพลงอื่นหรือผิดหมด → penalty < 0.2
    """
    if path_len <= 0:
        return 1.0

    per_frame = user_distance / path_len

    if   per_frame < 0.08:  penalty = 1.00
    elif per_frame < 0.12:  penalty = 0.90
    elif per_frame < 0.18:  penalty = 0.75
    elif per_frame < 0.25:  penalty = 0.55
    elif per_frame < 0.35:  penalty = 0.35
    elif per_frame < 0.50:  penalty = 0.20
    else:                   penalty = 0.10

    return float(penalty)


def grade_score(score: float) -> tuple:
    """แปลง score เป็น grade และ feedback"""
    if score >= 0.90:
        return "S", "🌟 Excellent! Very close to the reference!"
    elif score >= 0.80:
        return "A", "🎉 Great job! Most moves are correct."
    elif score >= 0.70:
        return "B", "👍 Good! Some moves need improvement."
    elif score >= 0.60:
        return "C", "💪 Fair. Keep practicing!"
    elif score >= 0.50:
        return "D", "🔄 Needs more practice on key moves."
    else:
        return "F", "📚 Please study and practice the choreography more."


# ========== Main Comparison Function ==========

def compare_dance(
    reference_data: dict,
    user_video_path: str,
    output_dir: str = "results",
    sample_fps: float = 10.0,
    create_comparison_video: bool = True,
    ref_start_offset: float = 0.0,
    user_start_offset: float = 0.0,
    pass_threshold: float = 0.5,
) -> dict:
    """
    เปรียบเทียบการเต้นของผู้ใช้กับ reference

    Args:
        reference_data: pose data จาก reference_processor
        user_video_path: path วิดีโอของผู้ใช้
        output_dir: โฟลเดอร์สำหรับบันทึกผลลัพธ์
        sample_fps: fps ที่ใช้ extract
        create_comparison_video: สร้างวิดีโอเปรียบเทียบหรือไม่
        ref_start_offset: ข้าม X วินาทีแรกของ reference (user เริ่มเร็วกว่า)
        user_start_offset: ข้าม X วินาทีแรกของ user video (user เริ่มช้ากว่า)

    Returns:
        dict ผลลัพธ์การเปรียบเทียบ
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*50)
    print("  🎵 K-Pop Dance Analyzer")
    print("="*50)

    # 1. Extract pose จากวิดีโอผู้ใช้
    print("\n[1/3] กำลัง extract pose จากวิดีโอของคุณ...")
    user_data = extract_pose_from_video(user_video_path, sample_fps=sample_fps)

    if len(user_data["keypoints"]) < 5:
        raise ValueError("วิดีโอของคุณสั้นเกินไป หรือหา pose ไม่เจอ กรุณาตรวจสอบแสงและมุมกล้อง")

    ref_kps  = reference_data["keypoints"]
    user_kps = user_data["keypoints"]

    # ตัด ref keypoints ตาม offset (ถ้า user เร็วกว่า ref)
    if ref_start_offset > 0:
        skip_frames = int(ref_start_offset * sample_fps)
        ref_kps = ref_kps[skip_frames:]
        ref_raw = reference_data.get("raw_keypoints", [])
        if ref_raw:
            reference_data = dict(reference_data)
            reference_data["raw_keypoints"] = ref_raw[skip_frames:]
        print(f"  ⏩ Skip reference {ref_start_offset:.2f}s ({skip_frames} frames)")

    # ตัด user keypoints ตาม offset (ถ้า user ช้ากว่า ref / เริ่มช้า)
    if user_start_offset > 0:
        skip_frames = int(user_start_offset * sample_fps)
        user_kps = user_kps[skip_frames:]
        user_raw = user_data.get("raw_keypoints", [])
        if user_raw:
            user_data = dict(user_data)
            user_data["raw_keypoints"] = user_raw[skip_frames:]
        print(f"  ⏩ Skip user video {user_start_offset:.2f}s ({skip_frames} frames)")

    # 2. DTW alignment
    print(f"\n[2/3] กำลังเปรียบเทียบท่าเต้น ({len(ref_kps)} vs {len(user_kps)} frames)...")
    dtw_distance, dtw_path = align_sequences_dtw(ref_kps, user_kps)

    # คำนวณ similarity scores
    raw_scores = compute_similarity_along_path(ref_kps, user_kps, dtw_path)
    smooth = smooth_scores(raw_scores)

    # ── DTW distance penalty ───────────────────────────────────────────
    # per-frame cosine distance บอกว่าท่าต่างกันแค่ไหนโดยเฉลี่ย
    # ถ้าสูง → ท่าต่างกันมาก (อาจเป็นคนละเพลง) → หักคะแนน
    dtw_penalty    = _compute_dtw_penalty(dtw_distance, 0, len(dtw_path))
    per_frame_dist = dtw_distance / max(len(dtw_path), 1)

    raw_score_mean = float(np.mean(raw_scores))

    # ── Pass/Fail classification ──────────────────────────────────────────
    # ทำก่อน grade เพราะ binary classifier ส่งผลต่อ overall_score
    pf_analysis = _analyze_pass_fail(user_kps, sample_fps, threshold=pass_threshold)

    # ถ้ามี binary classifier (supervised) → ใช้ pass_rate ปรับคะแนน
    # ถ้ามีแค่ anomaly detector → แสดงผลอย่างเดียว ไม่ปรับคะแนน
    if pf_analysis.get('available') and pf_analysis.get('model_type') == 'supervised':
        pass_rate = pf_analysis['pass_rate']
        if   pass_rate >= 0.85: pass_weight = 1.00
        elif pass_rate >= 0.65: pass_weight = 0.70
        elif pass_rate >= 0.45: pass_weight = 0.40
        else:                   pass_weight = 0.15
        overall_score = float(raw_score_mean * dtw_penalty * pass_weight)
        print(f"  Pass/Fail classifier: {pass_rate:.1%} PASS  →  weight {pass_weight:.2f}x")
    else:
        overall_score = float(raw_score_mean * dtw_penalty)

    print(f"  DTW per-frame distance: {per_frame_dist:.3f}  →  penalty {dtw_penalty:.2f}x")
    if dtw_penalty < 0.90:
        print(f"  ⚠️  ท่าต่างจาก reference มาก")

    grade, feedback = grade_score(overall_score)

    print(f"\n[3/3] สร้าง visualization...")

    # 3. Body part analysis (ต้องทำก่อน plot)
    body_analysis = _analyze_body_parts(ref_kps, user_kps, dtw_path)

    # 4. สร้าง plot
    plot_path = _create_score_plot(raw_scores, smooth, overall_score, grade, output_dir, body_analysis)

    # 5. สร้างวิดีโอเปรียบเทียบ (optional)
    video_path = None
    if create_comparison_video:
        video_path = _create_comparison_video(
            reference_data, user_data, dtw_path, raw_scores, output_dir,
            ref_start_offset=ref_start_offset,
            user_start_offset=user_start_offset,
        )

    results = {
        "overall_score": overall_score,
        "grade": grade,
        "feedback": feedback,
        "dtw_distance": float(dtw_distance),
        "frame_scores": raw_scores.tolist(),
        "smooth_scores": smooth.tolist(),
        "body_analysis": body_analysis,
        "pf_analysis":      pf_analysis,
        "pass_threshold":   pass_threshold,
        "output_plot": plot_path,
        "output_video": video_path,
        "num_ref_frames": len(ref_kps),
        "num_user_frames": len(user_kps),
    }

    _print_results(results)
    return results


def _analyze_body_parts(ref_kps: list, user_kps: list, dtw_path: list) -> dict:
    """วิเคราะห์ similarity แยกตาม body part"""

    # grouping ของ keypoints
    body_groups = {
        "แขนซ้าย": [
            DANCE_KEYPOINTS.index(11),  # left_shoulder
            DANCE_KEYPOINTS.index(13),  # left_elbow
            DANCE_KEYPOINTS.index(15),  # left_wrist
        ],
        "แขนขวา": [
            DANCE_KEYPOINTS.index(12),
            DANCE_KEYPOINTS.index(14),
            DANCE_KEYPOINTS.index(16),
        ],
        "ลำตัว": [
            DANCE_KEYPOINTS.index(11),
            DANCE_KEYPOINTS.index(12),
            DANCE_KEYPOINTS.index(23),
            DANCE_KEYPOINTS.index(24),
            DANCE_KEYPOINTS.index(0),  # nose
        ],
        "ขาซ้าย": [
            DANCE_KEYPOINTS.index(23),
            DANCE_KEYPOINTS.index(25),
            DANCE_KEYPOINTS.index(27),
        ],
        "ขาขวา": [
            DANCE_KEYPOINTS.index(24),
            DANCE_KEYPOINTS.index(26),
            DANCE_KEYPOINTS.index(28),
        ],
    }

    analysis = {}
    for part_name, indices in body_groups.items():
        part_scores = []
        for ri, ui in dtw_path:
            ref_part = ref_kps[ri][indices]
            user_part = user_kps[ui][indices]
            sim = compute_frame_similarity(ref_part, user_part)
            part_scores.append(sim)
        analysis[part_name] = {
            "score": float(np.mean(part_scores)),
            "grade": grade_score(float(np.mean(part_scores)))[0]
        }

    return analysis


def _analyze_pass_fail(user_kps: list, sample_fps: float = 10.0, threshold: float = 0.5) -> dict:
    """
    ตรวจจับ PASS/FAIL ต่อ frame
    ลำดับความสำคัญ:
      1. BinaryClassifier (supervised — ต้องการ FAIL video)  → ปรับคะแนนหลัก
      2. AnomalyDetector  (unsupervised — ไม่ต้องการ FAIL data) → แสดงผลอย่างเดียว
    """
    from pose_classifier import BinaryClassifier, AnomalyDetector

    kps_arr = np.array(user_kps)

    # ลอง binary classifier ก่อน
    bc = BinaryClassifier(sample_fps=sample_fps)
    if bc.is_loaded:
        try:
            frames     = bc.predict_frames(kps_arr, threshold=threshold)
            pass_count = sum(1 for f in frames if f['is_pass'])
            fail_count = len(frames) - pass_count
            pass_rate  = pass_count / max(len(frames), 1)
            return {
                'available':   True,
                'model_type':  'supervised',
                'frames':      frames,
                'pass_count':  pass_count,
                'fail_count':  fail_count,
                'pass_rate':   pass_rate,
            }
        except Exception as e:
            print(f"  BinaryClassifier error: {e}")

    # fallback → anomaly detector
    ad = _get_pf_classifier(sample_fps)
    if ad is None or not ad.is_loaded:
        return {'available': False}
    try:
        frames     = ad.predict_frames(kps_arr, smooth_window=5, threshold=threshold)
        pass_count = sum(1 for f in frames if f['is_pass'])
        fail_count = len(frames) - pass_count
        pass_rate  = pass_count / max(len(frames), 1)
        return {
            'available':   True,
            'model_type':  'anomaly',
            'frames':      frames,
            'pass_count':  pass_count,
            'fail_count':  fail_count,
            'pass_rate':   pass_rate,
        }
    except Exception as e:
        return {'available': False, 'error': str(e)}


def _create_score_plot(
    raw_scores: np.ndarray,
    smooth_scores: np.ndarray,
    overall_score: float,
    grade: str,
    output_dir: str,
    body_analysis: dict = None,
) -> str:
    """สร้างกราฟแสดงผล similarity score"""

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('#1a1a2e')

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ---- Plot 1: Frame-by-frame score ----
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#16213e')

    x = np.arange(len(raw_scores))
    ax1.fill_between(x, raw_scores, alpha=0.3, color='#00d4ff')
    ax1.plot(x, raw_scores, color='#00d4ff', alpha=0.5, linewidth=1, label='Raw Score')
    ax1.plot(x, smooth_scores, color='#ff6b9d', linewidth=2.5, label='Smoothed Score')
    ax1.axhline(y=overall_score, color='#ffd700', linewidth=1.5, linestyle='--',
                label=f'Average: {overall_score:.1%}')

    ax1.set_title('Similarity Score Over Time', color='white', fontsize=13, pad=10)
    ax1.set_xlabel('Frame', color='#aaaaaa')
    ax1.set_ylabel('Similarity', color='#aaaaaa')
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(colors='#aaaaaa')
    ax1.spines['bottom'].set_color('#444444')
    ax1.spines['left'].set_color('#444444')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(facecolor='#16213e', labelcolor='white', fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # ---- Plot 2: Overall Score Gauge ----
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#16213e')

    # Gauge chart
    score_pct = overall_score
    colors_gradient = ['#ff4444', '#ff8800', '#ffcc00', '#88cc00', '#00cc44']
    grade_colors = {'F': '#ff4444', 'D': '#ff8800', 'C': '#ffcc00', 'B': '#88cc00', 'A': '#00cc44', 'S': '#00eeff'}

    theta = np.linspace(np.pi, 0, 100)
    for i in range(len(theta)-1):
        color_idx = int(i / len(theta) * len(colors_gradient))
        ax2.plot([np.cos(theta[i]), np.cos(theta[i+1])],
                 [np.sin(theta[i]), np.sin(theta[i+1])],
                 color=colors_gradient[min(color_idx, len(colors_gradient)-1)],
                 linewidth=12, solid_capstyle='round')

    # Needle
    needle_angle = np.pi * (1 - score_pct)
    ax2.annotate('', xy=(np.cos(needle_angle)*0.7, np.sin(needle_angle)*0.7),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='white', lw=2))

    ax2.text(0, -0.2, f'{score_pct:.1%}', ha='center', va='center',
             fontsize=22, fontweight='bold', color=grade_colors.get(grade, 'white'))
    ax2.text(0, -0.45, f'Grade: {grade}', ha='center', va='center',
             fontsize=14, color=grade_colors.get(grade, 'white'))

    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-0.6, 1.3)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Overall Score', color='white', fontsize=12, pad=8)

    # ---- Plot 3: Body Part Analysis (radar chart) ----
    ax3 = fig.add_subplot(gs[1, 1], polar=True)
    ax3.set_facecolor('#16213e')

    if body_analysis:
        parts = list(body_analysis.keys())
        scores = [body_analysis[p]["score"] for p in parts]
        N = len(parts)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        scores_plot = scores + scores[:1]

        ax3.set_theta_offset(np.pi / 2)
        ax3.set_theta_direction(-1)
        ax3.plot(angles, scores_plot, 'o-', linewidth=2, color='#00d4ff')
        ax3.fill(angles, scores_plot, alpha=0.25, color='#00d4ff')
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(parts, color='white', fontsize=9)
        ax3.set_ylim(0, 1)
        ax3.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax3.set_yticklabels(['25%','50%','75%','100%'], color='#888888', fontsize=7)
        ax3.grid(color='#444444', alpha=0.5)
        ax3.spines['polar'].set_color('#444444')

    ax3.set_title('Body Part Scores', color='white', fontsize=12, pad=15)

    plt.suptitle('🎵 K-Pop Dance Analysis Report', color='white', fontsize=15,
                 fontweight='bold', y=0.98)

    plot_path = os.path.join(output_dir, "dance_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()

    return plot_path



def _extract_audio(video_path: str, out_wav: str) -> bool:
    """Extract audio จากวิดีโอเป็น WAV ด้วย ffmpeg"""
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "22050", "-ac", "1", out_wav
        ], capture_output=True, check=True)
        return os.path.exists(out_wav)
    except Exception:
        return False


def compute_beat_sync_map(ref_video: str, user_video: str, verbose: bool = True) -> np.ndarray:
    """
    คำนวณ beat-based time mapping จาก user → reference

    Returns:
        beat_map: array ขนาด (N,) โดย beat_map[i] = เวลาใน user video
                  ที่ตรงกับวินาทีที่ i ของ reference video
                  ถ้าคำนวณไม่ได้ returns None
    """
    try:
        import librosa

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_wav  = os.path.join(tmpdir, "ref.wav")
            user_wav = os.path.join(tmpdir, "user.wav")

            if verbose:
                print("  🎵 Extracting audio...")

            ok_r = _extract_audio(ref_video, ref_wav)
            ok_u = _extract_audio(user_video, user_wav)

            if not ok_r or not ok_u:
                print("  ⚠️ ไม่สามารถ extract audio ได้ ใช้ linear sync แทน")
                return None

            if verbose:
                print("  🥁 Detecting beats...")

            y_ref,  sr = librosa.load(ref_wav,  sr=22050)
            y_user, _  = librosa.load(user_wav, sr=22050)

            # beat tracking
            tempo_r, beats_r = librosa.beat.beat_track(y=y_ref,  sr=sr, units='time')
            tempo_u, beats_u = librosa.beat.beat_track(y=y_user, sr=sr, units='time')

            if len(beats_r) < 4 or len(beats_u) < 4:
                print("  ⚠️ หา beat ได้น้อยเกินไป ใช้ linear sync แทน")
                return None

            if verbose:
                print(f"  Reference: {len(beats_r)} beats @ {float(tempo_r):.1f} BPM")
                print(f"  User     : {len(beats_u)} beats @ {float(tempo_u):.1f} BPM")

            # สร้าง time map: สำหรับแต่ละวินาทีใน ref → วินาทีที่ตรงใน user
            ref_duration  = len(y_ref)  / sr
            user_duration = len(y_user) / sr

            # ใช้ DTW บน beat timestamps เพื่อ align
            from fastdtw import fastdtw as _dtw
            beats_r2 = beats_r.reshape(-1, 1)
            beats_u2 = beats_u.reshape(-1, 1)
            _, beat_path = _dtw(beats_r2, beats_u2, dist=lambda a, b: abs(a[0] - b[0]))

            # สร้าง lookup: ref_beat_time → user_beat_time
            ref_beat_times  = [beats_r[ri] for ri, ui in beat_path]
            user_beat_times = [beats_u[ui] for ri, ui in beat_path]

            # interpolate เป็น per-second map
            n_seconds = int(ref_duration) + 1
            beat_map  = np.interp(
                np.arange(n_seconds, dtype=float),
                ref_beat_times,
                user_beat_times,
                left=0.0,
                right=user_duration,
            )

            if verbose:
                print(f"  ✅ Beat sync map สร้างสำเร็จ ({n_seconds}s)")

            return beat_map

    except ImportError:
        print("  ⚠️ ไม่พบ librosa ใช้ linear sync แทน")
        return None
    except Exception as e:
        print(f"  ⚠️ Beat sync error: {e} — ใช้ linear sync แทน")
        return None

def _create_comparison_video(
    reference_data: dict,
    user_data: dict,
    dtw_path: list,
    frame_scores: np.ndarray,
    output_dir: str,
    max_frames: int = 99999,
    ref_start_offset: float = 0.0,
    user_start_offset: float = 0.0,
) -> Optional[str]:
    """
    สร้างวิดีโอเปรียบเทียบ side-by-side
    - sync วิดีโอตาม offset ที่ตั้งไว้
    - skeleton ใช้ DTW path จริง (ไม่ใช่ linear ratio)
    """
    try:
        ref_video_path  = reference_data.get("video_info", {}).get("path")
        user_video_path = user_data.get("video_info", {}).get("path")

        if not ref_video_path or not os.path.exists(ref_video_path):
            print("  ⚠️ ไม่พบ reference video สำหรับสร้าง comparison video")
            return None

        ref_kps  = reference_data["keypoints"]
        user_kps = user_data["keypoints"]
        ref_raw  = reference_data.get("raw_keypoints", [])
        user_raw = user_data.get("raw_keypoints", [])

        # ── DTW lookup: ref sample index → user sample index ──────────
        # dtw_path เป็น [(ri, ui), ...] หลังตัด offset แล้ว
        ref_to_user_si: dict[int, int] = {}
        for ri, ui in dtw_path:
            if ri not in ref_to_user_si:
                ref_to_user_si[ri] = ui

        # score ต่อ ref sample index
        score_map: dict[int, float] = {}
        for path_idx, (ri, ui) in enumerate(dtw_path):
            if ri not in score_map:
                score_map[ri] = frame_scores[path_idx] if path_idx < len(frame_scores) else 0.5

        cap_ref  = cv2.VideoCapture(ref_video_path)
        cap_user = cv2.VideoCapture(user_video_path)

        ref_fps   = cap_ref.get(cv2.CAP_PROP_FPS)  or 30.0
        user_fps  = cap_user.get(cv2.CAP_PROP_FPS) or 30.0
        frame_w   = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h   = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ref_sample_fps  = reference_data.get("sample_fps", 10.0)
        user_sample_fps = user_data.get("sample_fps", 10.0)
        sample_interval = max(1, int(ref_fps / ref_sample_fps))   # กี่ video-frame ต่อ 1 sample

        # ── Skip ตาม offset ───────────────────────────────────────────
        if ref_start_offset > 0:
            skip_r = int(ref_start_offset * ref_fps)
            cap_ref.set(cv2.CAP_PROP_POS_FRAMES, skip_r)
            print(f"  ⏩ Skip reference {ref_start_offset:.2f}s ({skip_r} frames)")

        if user_start_offset > 0:
            skip_u = int(user_start_offset * user_fps)
            cap_user.set(cv2.CAP_PROP_POS_FRAMES, skip_u)
            print(f"  ⏩ Skip user video {user_start_offset:.2f}s ({skip_u} frames)")

        out_w = frame_w * 2 + 6
        out_h = frame_h
        output_path = os.path.join(output_dir, "comparison.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(output_path, fourcc, ref_fps, (out_w, out_h))

        # Beat sync map (optional)
        beat_map      = compute_beat_sync_map(ref_video_path, user_video_path)
        use_beat_sync = beat_map is not None
        sync_mode     = "beat sync 🥁" if use_beat_sync else "linear sync"
        print(f"  กำลังสร้าง comparison video ({sync_mode})...")

        ref_frame_idx  = int(ref_start_offset * ref_fps)   # video frame index (ใน file)
        user_frame_idx = int(user_start_offset * user_fps)  # video frame index (ใน file)
        sample_count   = 0    # ref sample index (หลัง offset)
        written        = 0
        current_score  = 0.5
        last_frame_u   = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        # skeleton ที่วาดล่าสุด (persist ข้ามเฟรมวิดีโอที่ไม่ใช่ sample point)
        last_frame_r_sk = None
        last_frame_u_sk = None

        while cap_ref.isOpened() and written < max_frames:
            ret_r, frame_r = cap_ref.read()
            if not ret_r:
                break

            # ── sync user video frame ตาม WALL CLOCK (linear) ──────────
            # video เล่นตามเวลาจริง → user ไม่ถูกเร่ง/ช้า
            # ถ้า timing ไม่ตรง ให้ปรับ USER_START_OFFSET ใน Cell 3
            elapsed_frames = ref_frame_idx - int(ref_start_offset * ref_fps)
            elapsed_sec    = elapsed_frames / ref_fps
            user_target_frame = int(user_start_offset * user_fps) + int(elapsed_sec * user_fps)

            while user_frame_idx <= user_target_frame:
                ret_u, frame_u_read = cap_user.read()
                if not ret_u:
                    break
                last_frame_u = frame_u_read
                user_frame_idx += 1

            frame_u = cv2.resize(last_frame_u, (frame_w, frame_h))

            # ── วาด skeleton เมื่อถึง sample point ────────────────────
            if elapsed_frames % sample_interval == 0:
                if sample_count in score_map:
                    current_score = score_map[sample_count]

                # ref skeleton
                if sample_count < len(ref_kps):
                    r_raw = ref_raw[sample_count] if sample_count < len(ref_raw) else None
                    frame_r = visualize_pose_on_frame(
                        frame_r, ref_kps[sample_count],
                        color=(0, 255, 100), raw_keypoints=r_raw)
                    last_frame_r_sk = (ref_kps[sample_count], r_raw)

                # user skeleton — linear time (ตรงกับ video ที่เห็น)
                # ดู ref sample ไหน → ดู user sample เดียวกัน
                # เห็นตรงๆ ว่าช้า/เร็ว/ผิดต่างกันยังไง
                user_si = min(sample_count, len(user_kps) - 1)
                u_raw = user_raw[user_si] if user_si < len(user_raw) else None
                frame_u = visualize_pose_on_frame(
                    frame_u, user_kps[user_si],
                    color=(255, 100, 0), raw_keypoints=u_raw)
                last_frame_u_sk = (user_kps[user_si], u_raw)

                sample_count += 1
            else:
                # ระหว่าง sample points → วาด skeleton เดิมซ้ำ ให้ smooth
                if last_frame_r_sk is not None:
                    frame_r = visualize_pose_on_frame(
                        frame_r, last_frame_r_sk[0],
                        color=(0, 255, 100), raw_keypoints=last_frame_r_sk[1])
                if last_frame_u_sk is not None:
                    frame_u = visualize_pose_on_frame(
                        frame_u, last_frame_u_sk[0],
                        color=(255, 100, 0), raw_keypoints=last_frame_u_sk[1])

            # ── overlay ────────────────────────────────────────────────
            score_color = _score_to_color(current_score)
            _draw_score_bar(frame_r, current_score, score_color, label="Artist")
            _draw_score_bar(frame_u, current_score, score_color, label="You")

            cv2.putText(frame_r, "REFERENCE", (10, frame_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2, cv2.LINE_AA)
            cv2.putText(frame_u, "YOU", (10, frame_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 2, cv2.LINE_AA)

            divider  = np.full((frame_h, 6, 3), 40, dtype=np.uint8)
            combined = np.hstack([frame_r, divider, frame_u])
            out.write(combined)
            written       += 1
            ref_frame_idx += 1

        cap_ref.release()
        cap_user.release()
        out.release()

        print(f"  ✓ บันทึก comparison video: {output_path} ({written} frames)")
        return output_path

    except Exception as e:
        import traceback
        print(f"  ⚠️ ไม่สามารถสร้าง comparison video: {e}")
        traceback.print_exc()
        return None


def _score_to_color(score: float) -> tuple:
    """แปลง score เป็นสี BGR"""
    if score >= 0.8:
        return (0, 220, 0)    # เขียว
    elif score >= 0.6:
        return (0, 200, 255)  # เหลือง
    else:
        return (0, 0, 220)    # แดง


def _draw_score_bar(frame: np.ndarray, score: float, color: tuple, label: str = "") -> None:
    """วาด score bar ที่มุมบนขวาของ frame"""
    h, w = frame.shape[:2]
    bar_x = w - 120
    bar_y = 15
    bar_w = 100
    bar_h = 18

    # Background
    cv2.rectangle(frame, (bar_x - 5, bar_y - 5), (bar_x + bar_w + 5, bar_y + bar_h + 20),
                  (0, 0, 0), -1)

    # Score bar
    fill_w = int(bar_w * score)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)

    # Text
    cv2.putText(frame, f"{label}: {score:.0%}", (bar_x, bar_y + bar_h + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _print_results(results: dict) -> None:
    """แสดงผลสรุปในรูปแบบสวยงาม"""
    grade_colors = {
        'S': '\033[96m', 'A': '\033[92m', 'B': '\033[93m',
        'C': '\033[93m', 'D': '\033[91m', 'F': '\033[91m'
    }
    RESET = '\033[0m'
    grade = results['grade']
    color = grade_colors.get(grade, '')

    print("\n" + "="*50)
    print(f"  📊 ผลการวิเคราะห์ท่าเต้น")
    print("="*50)
    print(f"  Overall Score: {color}{results['overall_score']:.1%}  (Grade: {grade}){RESET}")
    print(f"  {results['feedback']}")
    print("\n  Body Part Analysis:")
    for part, data in results["body_analysis"].items():
        bar = "█" * int(data["score"] * 20) + "░" * (20 - int(data["score"] * 20))
        print(f"    {part:8s}  [{bar}]  {data['score']:.0%}  ({data['grade']})")
    # Anomaly Detection analysis
    pf = results.get("pf_analysis", {})
    if pf.get("available"):
        GREEN = '\033[92m'; RED = '\033[91m'
        thresh = results.get('pass_threshold', 0.5)
        print(f"\n  Anomaly Detection (Isolation Forest):")
        print(f"    PASS rate  : {pf['pass_rate']:.1%}  "
              f"({pf['pass_count']} frames PASS / {pf['fail_count']} frames FAIL)")
        print(f"    Threshold  : {thresh}")
        # แสดง FAIL runs ที่ยาวกว่า 2 วินาที
        frames = pf.get('frames', [])
        fail_runs = []
        in_fail = False
        for f in frames:
            if not f['is_pass'] and not in_fail:
                run_start = f['time']; in_fail = True
            elif f['is_pass'] and in_fail:
                fail_runs.append((run_start, f['time'])); in_fail = False
        if in_fail and frames:
            fail_runs.append((run_start, frames[-1]['time']))
        fail_runs = [(s, e) for s, e in fail_runs if e - s >= 2.0]
        if fail_runs:
            print(f"\n    ช่วงที่ต้องฝึกเพิ่ม:")
            for s, e in fail_runs[:8]:
                ms, ss = divmod(int(s), 60)
                me, se = divmod(int(e), 60)
                print(f"      {ms:02d}:{ss:02d} – {me:02d}:{se:02d}  ({e-s:.1f}s)")

    print()
    if results.get("output_plot"):
        print(f"  Plot : {results['output_plot']}")
    if results.get("output_video"):
        print(f"  Video: {results['output_video']}")
    print("="*50)


# ========== Quick-start helper ==========

def quick_compare(
    youtube_url: str,
    user_video_path: str,
    start_time: str = None,
    end_time: str = None,
    output_dir: str = "results",
) -> dict:
    """
    ฟังก์ชัน all-in-one สำหรับเปรียบเทียบแบบรวดเร็ว

    Args:
        youtube_url: URL ของวิดีโอ reference บน YouTube
        user_video_path: path วิดีโอของผู้ใช้
        start_time: เวลาเริ่มต้นของ reference (เช่น "0:30")
        end_time: เวลาสิ้นสุดของ reference (เช่น "1:00")
        output_dir: โฟลเดอร์สำหรับบันทึกผลลัพธ์

    Returns:
        dict ผลการเปรียบเทียบ
    """
    from reference_processor import download_youtube_video

    # ดาวน์โหลด reference
    ref_video = download_youtube_video(
        youtube_url,
        output_dir=os.path.join(output_dir, "reference_videos"),
        start_time=start_time,
        end_time=end_time,
    )

    # Extract reference poses
    print("\nกำลัง extract pose จาก reference video...")
    ref_data = extract_pose_from_video(ref_video, sample_fps=10)
    ref_data["video_info"]["path"] = ref_video

    # Compare
    return compare_dance(ref_data, user_video_path, output_dir=output_dir)


if __name__ == "__main__":
    print("=== Dance Comparator ===")
    print("ดู Jupyter Notebook สำหรับการใช้งาน step-by-step")
