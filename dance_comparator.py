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

from reference_processor import (
    extract_pose_from_video,
    save_pose_data,
    load_pose_data,
    visualize_pose_on_frame,
    DANCE_KEYPOINTS,
    KEYPOINT_NAMES,
)


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
    จัดเรียง sequence สอง ชุดด้วย Dynamic Time Warping
    เพื่อรับมือกับความเร็วในการเต้นที่ต่างกัน

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


def grade_score(score: float) -> tuple:
    """แปลง score เป็น grade และ feedback"""
    if score >= 0.90:
        return "S", "🌟 ยอดเยี่ยมมาก! ท่าทางใกล้เคียงศิลปินมาก"
    elif score >= 0.80:
        return "A", "🎉 เก่งมาก! ท่าทางส่วนใหญ่ถูกต้อง"
    elif score >= 0.70:
        return "B", "👍 ดีครับ! มีบางท่าที่ต้องปรับปรุง"
    elif score >= 0.60:
        return "C", "💪 พอใช้ได้ ลองฝึกซ้ำอีกหน่อยนะครับ"
    elif score >= 0.50:
        return "D", "🔄 ยังต้องฝึกอีก โดยเฉพาะท่วงท่าหลัก"
    else:
        return "F", "📚 ยังต้องดูและฝึกท่าเต้นเพิ่มเติมครับ"


# ========== Main Comparison Function ==========

def compare_dance(
    reference_data: dict,
    user_video_path: str,
    output_dir: str = "results",
    sample_fps: float = 10.0,
    create_comparison_video: bool = True,
) -> dict:
    """
    เปรียบเทียบการเต้นของผู้ใช้กับ reference

    Args:
        reference_data: pose data จาก reference_processor
        user_video_path: path วิดีโอของผู้ใช้
        output_dir: โฟลเดอร์สำหรับบันทึกผลลัพธ์
        sample_fps: fps ที่ใช้ extract
        create_comparison_video: สร้างวิดีโอเปรียบเทียบหรือไม่

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

    ref_kps = reference_data["keypoints"]
    user_kps = user_data["keypoints"]

    # 2. DTW alignment
    print(f"\n[2/3] กำลังเปรียบเทียบท่าเต้น ({len(ref_kps)} vs {len(user_kps)} frames)...")
    dtw_distance, dtw_path = align_sequences_dtw(ref_kps, user_kps)

    # คำนวณ similarity scores
    raw_scores = compute_similarity_along_path(ref_kps, user_kps, dtw_path)
    smooth = smooth_scores(raw_scores)

    overall_score = float(np.mean(raw_scores))
    grade, feedback = grade_score(overall_score)

    print(f"\n[3/3] สร้าง visualization...")

    # 3. สร้าง plot
    plot_path = _create_score_plot(raw_scores, smooth, overall_score, grade, output_dir)

    # 4. สร้างวิดีโอเปรียบเทียบ (optional)
    video_path = None
    if create_comparison_video:
        video_path = _create_comparison_video(
            reference_data, user_data, dtw_path, raw_scores, output_dir
        )

    # 5. Body part analysis
    body_analysis = _analyze_body_parts(ref_kps, user_kps, dtw_path)

    results = {
        "overall_score": overall_score,
        "grade": grade,
        "feedback": feedback,
        "dtw_distance": float(dtw_distance),
        "frame_scores": raw_scores.tolist(),
        "smooth_scores": smooth.tolist(),
        "body_analysis": body_analysis,
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


def _create_score_plot(
    raw_scores: np.ndarray,
    smooth_scores: np.ndarray,
    overall_score: float,
    grade: str,
    output_dir: str,
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

    ax1.set_title('Similarity Score ตลอดการเต้น', color='white', fontsize=13, pad=10)
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

    # ---- Plot 3: Body Part Analysis ----
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#16213e')

    # Placeholder - will be filled after body analysis
    ax3.text(0.5, 0.5, 'Body Part\nAnalysis\n(ดูใน results dict)',
             ha='center', va='center', color='#aaaaaa', fontsize=10,
             transform=ax3.transAxes)
    ax3.set_title('Body Part Scores', color='white', fontsize=12, pad=8)
    ax3.axis('off')

    plt.suptitle('🎵 K-Pop Dance Analysis Report', color='white', fontsize=15,
                 fontweight='bold', y=0.98)

    plot_path = os.path.join(output_dir, "dance_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()

    return plot_path


def _create_comparison_video(
    reference_data: dict,
    user_data: dict,
    dtw_path: list,
    frame_scores: np.ndarray,
    output_dir: str,
    max_frames: int = 300,
) -> Optional[str]:
    """สร้างวิดีโอเปรียบเทียบ side-by-side"""

    try:
        ref_video_path = reference_data.get("video_info", {}).get("path")
        user_video_path = user_data.get("video_info", {}).get("path")

        if not ref_video_path or not os.path.exists(ref_video_path):
            print("  ⚠️ ไม่พบ reference video สำหรับสร้าง comparison video")
            return None

        ref_kps = reference_data["keypoints"]
        user_kps = user_data["keypoints"]

        cap_ref = cv2.VideoCapture(ref_video_path)
        cap_user = cv2.VideoCapture(user_video_path)

        ref_fps = cap_ref.get(cv2.CAP_PROP_FPS)
        frame_w = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
        user_w = int(cap_user.get(cv2.CAP_PROP_FRAME_WIDTH))
        user_h = int(cap_user.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Resize user video ให้ตรงกับ reference
        out_w = frame_w * 2 + 10  # side by side
        out_h = frame_h

        output_path = os.path.join(output_dir, "comparison.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, ref_fps, (out_w, out_h))

        # Map DTW path: ref_frame_idx → (score, user_frame_idx)
        path_dict = {}
        for path_idx, (ri, ui) in enumerate(dtw_path):
            if ri not in path_dict:
                path_dict[ri] = (frame_scores[path_idx] if path_idx < len(frame_scores) else 0.5, ui)

        ref_sample_fps = reference_data.get("sample_fps", 10.0)
        frame_interval = max(1, int(ref_fps / ref_sample_fps))

        ref_frame_count = 0
        sample_count = 0

        print(f"  กำลังสร้าง comparison video...")
        written_frames = 0

        while cap_ref.isOpened() and written_frames < max_frames * frame_interval:
            ret_r, frame_r = cap_ref.read()
            if not ret_r:
                break

            if ref_frame_count % frame_interval == 0 and sample_count in path_dict:
                score, user_sample_idx = path_dict[sample_count]

                # ดึง frame จาก user video
                user_frame_pos = int(user_sample_idx * frame_interval)
                cap_user.set(cv2.CAP_PROP_POS_FRAMES, user_frame_pos)
                ret_u, frame_u = cap_user.read()

                if not ret_u:
                    frame_u = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

                # Resize user frame
                frame_u_resized = cv2.resize(frame_u, (frame_w, frame_h))

                # วาด skeleton บน frames
                if sample_count < len(ref_kps):
                    frame_r = visualize_pose_on_frame(frame_r, ref_kps[sample_count], color=(0, 255, 100))
                if user_sample_idx < len(user_kps):
                    frame_u_resized = visualize_pose_on_frame(frame_u_resized, user_kps[user_sample_idx], color=(255, 100, 0))

                # สร้าง score bar
                score_color = _score_to_color(score)
                _draw_score_bar(frame_r, score, score_color, label="ศิลปิน")
                _draw_score_bar(frame_u_resized, score, score_color, label="คุณ")

                # รวม side-by-side
                divider = np.zeros((frame_h, 10, 3), dtype=np.uint8)
                combined = np.hstack([frame_r, divider, frame_u_resized])
                out.write(combined)
                written_frames += 1

                sample_count += 1

            ref_frame_count += 1

        cap_ref.release()
        cap_user.release()
        out.release()

        print(f"  ✓ บันทึก comparison video: {output_path}")
        return output_path

    except Exception as e:
        print(f"  ⚠️ ไม่สามารถสร้าง comparison video: {e}")
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
    print()
    if results.get("output_plot"):
        print(f"  📈 Plot: {results['output_plot']}")
    if results.get("output_video"):
        print(f"  🎬 Video: {results['output_video']}")
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
