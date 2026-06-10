"""
Vrikshasana (Tree Pose) Analyzer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Detects ONLY Vrikshasana using exact reference joint angles & 3D coordinates
• ALL 33 MediaPipe landmarks must be visible (visibility > 0.5) before any
  analysis is shown — until then a clear prompt is displayed.
• Real-time per-joint scoring + post-session written report.
• MIRROR POSE SUPPORT: both left-leg-raised and right-leg-raised variants
  are scored equally — whichever orientation matches better is used.

Reference data captured from a correct Vrikshasana posture:
  Angles  — left_knee:27.19°  right_knee:175.10°  left_hip:126.03°
             right_hip:177.81°  left_elbow:176.34°  right_elbow:169.60°
             left_shoulder:174.51°  right_shoulder:168.42°  spine:180.00°
  Coords  — 33 world-space landmarks provided by user.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import sys
from datetime import datetime
from pathlib import Path

# ─── MediaPipe ────────────────────────────────────────────────────────────────
mp_pose           = mp.solutions.pose
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ─── Colors (BGR) ─────────────────────────────────────────────────────────────
C = {
    "green":  (70,  200,  70),
    "yellow": (30,  210, 230),
    "red":    (55,   75, 220),
    "white":  (240, 240, 240),
    "gray":   (150, 150, 150),
    "accent": (50,  180, 240),
    "dark":   (18,   18,  18),
    "orange": (40,  140, 255),
}

# ─── Landmark indices ─────────────────────────────────────────────────────────
KP = {
    "nose":             0,
    "left_shoulder":   11, "right_shoulder":  12,
    "left_elbow":      13, "right_elbow":     14,
    "left_wrist":      15, "right_wrist":     16,
    "left_hip":        23, "right_hip":       24,
    "left_knee":       25, "right_knee":      26,
    "left_ankle":      27, "right_ankle":     28,
    "left_heel":       29, "right_heel":      30,
    "left_foot_index": 31, "right_foot_index":32,
}

# ══════════════════════════════════════════════════════════════════════════════
#  VRIKSHASANA REFERENCE ANGLES  (from correct posture capture)
#  Format:  joint_name : (ideal_angle°,  tolerance°)
# ══════════════════════════════════════════════════════════════════════════════
VRIKS_REF = {
    "left_knee":      (27.19,  12.0),   # raised leg — deep bend
    "right_knee":     (175.10,  8.0),   # standing leg — fully extended
    "left_hip":       (126.03, 18.0),   # raised-leg hip open
    "right_hip":      (177.81, 10.0),   # standing-leg hip straight
    "left_elbow":     (176.34, 12.0),   # arms extended overhead
    "right_elbow":    (169.60, 12.0),
    "left_shoulder":  (174.51, 12.0),   # shoulders raised/aligned
    "right_shoulder": (168.42, 12.0),
    "spine":          (180.00, 10.0),   # perfectly vertical
}

# ── Reference 3D world-space coordinates (33 landmarks) ──────────────────────
VRIKS_COORDS = {
     0: [-0.2045, -2.7060, -0.0012],
     1: [-0.1408, -2.8212, -0.0010],
     2: [-0.0978, -2.8195, -0.0010],
     3: [-0.0529, -2.8170, -0.0010],
     4: [-0.2574, -2.8098, -0.0011],
     5: [-0.3037, -2.8017, -0.0011],
     6: [-0.3454, -2.7939, -0.0011],
     7: [ 0.0132, -2.7444, -0.0005],
     8: [-0.4097, -2.7212, -0.0006],
     9: [-0.1353, -2.5476, -0.0010],
    10: [-0.2685, -2.5486, -0.0010],
    11: [ 0.3181, -2.2582, -0.0005],   # left_shoulder
    12: [-0.6709, -2.1099, -0.0006],   # right_shoulder
    13: [ 0.2023, -3.3259, -0.0008],   # left_elbow
    14: [-0.6142, -3.2053, -0.0012],   # right_elbow
    15: [ 0.0228, -4.3599, -0.0010],   # left_wrist
    16: [-0.3791, -4.1952, -0.0014],   # right_wrist
    17: [ 0.0016, -4.5362, -0.0012],
    18: [-0.2745, -4.4540, -0.0015],
    19: [-0.0293, -4.5443, -0.0012],
    20: [-0.2588, -4.4644, -0.0015],
    21: [-0.0409, -4.4909, -0.0011],
    22: [-0.2697, -4.4056, -0.0014],
    23: [ 0.3455, -0.0357,  0.0001],   # left_hip
    24: [-0.3455,  0.0357, -0.0001],   # right_hip
    25: [ 1.5610,  0.8257, -0.0006],   # left_knee  (raised leg, swings laterally)
    26: [-0.1631,  1.6528, -0.0001],   # right_knee (standing leg)
    27: [ 0.0039,  0.6031,  0.0008],   # left_ankle
    28: [ 0.1320,  3.1247,  0.0009],   # right_ankle
    29: [-0.1472,  0.4708,  0.0009],   # left_heel
    30: [ 0.1950,  3.2654,  0.0010],   # right_heel
    31: [-0.2206,  1.1126,  0.0007],   # left_foot_index
    32: [-0.1067,  3.6978,  0.0004],   # right_foot_index
}


# ══════════════════════════════════════════════════════════════════════════════
#  VISIBILITY GATE — ALL 33 landmarks must be above threshold
# ══════════════════════════════════════════════════════════════════════════════
VIS_THRESHOLD = 0.50

def all_landmarks_visible(landmarks):
    """Return (all_ok: bool, missing_count: int)."""
    missing = sum(1 for lm in landmarks if lm.visibility < VIS_THRESHOLD)
    return missing == 0, missing


# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def angle_between(a, b, c):
    """Angle (°) at joint b formed by points a–b–c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cos_a   = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

def vertical_angle(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return float(abs(np.degrees(np.arctan2(dx, -dy))))

def lm_xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)


# ══════════════════════════════════════════════════════════════════════════════
#  METRIC EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def extract_metrics(landmarks, w, h):
    g = lambda idx: lm_xy(landmarks, idx, w, h)

    nose  = g(KP["nose"])
    l_sh  = g(KP["left_shoulder"]);  r_sh  = g(KP["right_shoulder"])
    l_el  = g(KP["left_elbow"]);     r_el  = g(KP["right_elbow"])
    l_wr  = g(KP["left_wrist"]);     r_wr  = g(KP["right_wrist"])
    l_hip = g(KP["left_hip"]);       r_hip = g(KP["right_hip"])
    l_kn  = g(KP["left_knee"]);      r_kn  = g(KP["right_knee"])
    l_an  = g(KP["left_ankle"]);     r_an  = g(KP["right_ankle"])

    mid_hip = ((l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2)
    mid_sh  = ((l_sh[0]+r_sh[0])/2,   (l_sh[1]+r_sh[1])/2)
    # downward reference point for spine verticality
    down    = (mid_hip[0], mid_hip[1] + 100)

    return {
        "left_knee":      angle_between(l_hip,  l_kn,  l_an),
        "right_knee":     angle_between(r_hip,  r_kn,  r_an),
        "left_hip":       angle_between(l_sh,   l_hip, l_kn),
        "right_hip":      angle_between(r_sh,   r_hip, r_kn),
        "left_elbow":     angle_between(l_sh,   l_el,  l_wr),
        "right_elbow":    angle_between(r_sh,   r_el,  r_wr),
        "left_shoulder":  angle_between(l_el,   l_sh,  l_hip),
        "right_shoulder": angle_between(r_el,   r_sh,  r_hip),
        "spine":          angle_between(mid_sh, mid_hip, down),
        # auxiliary metrics for feedback
        "hip_tilt_deg":   abs(l_hip[1] - r_hip[1]) / (h * 0.01),
        "neck_deg":       vertical_angle(mid_sh, nose),
        "wrist_h_l":      (l_sh[1] - l_wr[1]) / h,   # +ve = wrist above shoulder
        "wrist_h_r":      (r_sh[1] - r_wr[1]) / h,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MIRROR METRICS — swap left↔right for mirror-image pose support
# ══════════════════════════════════════════════════════════════════════════════
def mirror_metrics(m):
    """
    Return a copy of metrics with all left/right pairs swapped.
    This allows someone doing the pose with the opposite leg raised
    (mirror image of the reference) to score equally well.
    """
    swaps = [
        ("left_knee",     "right_knee"),
        ("left_hip",      "right_hip"),
        ("left_elbow",    "right_elbow"),
        ("left_shoulder", "right_shoulder"),
        ("wrist_h_l",     "wrist_h_r"),
    ]
    mm = m.copy()
    for a, b in swaps:
        mm[a], mm[b] = m[b], m[a]
    return mm


# ══════════════════════════════════════════════════════════════════════════════
#  POSE DETECTION — Vrikshasana only
# ══════════════════════════════════════════════════════════════════════════════
def detect_vrikshasana(m):
    """Returns confidence 0-1 based on Vrikshasana-specific joint signatures."""
    score = 0.0

    # 1. Large asymmetry between knee angles (tree pose hallmark)
    knee_diff = abs(m["right_knee"] - m["left_knee"])
    if knee_diff > 120:  score += 0.35
    elif knee_diff > 80: score += 0.20

    # 2. Standing knee near reference 175°
    if abs(m["right_knee"] - 175.10) < 12: score += 0.20

    # 3. Raised knee near reference 27°
    if abs(m["left_knee"] - 27.19) < 18:   score += 0.20

    # 4. Arms raised overhead (wrists clearly above shoulders)
    if m["wrist_h_l"] > 0.05 and m["wrist_h_r"] > 0.05: score += 0.25

    return min(score, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING — weighted deviation from reference angles
# ══════════════════════════════════════════════════════════════════════════════
JOINT_WEIGHTS = {
    "right_knee":     20,   # standing leg — most critical
    "left_knee":      18,   # raised leg
    "spine":          15,
    "right_hip":      12,
    "left_hip":       10,
    "left_shoulder":   8,
    "right_shoulder":  8,
    "left_elbow":      5,
    "right_elbow":     4,
}
TOTAL_WEIGHT = sum(JOINT_WEIGHTS.values())  # 100

def compute_score(m):
    """Overall score 0-100 via weighted joint deviations."""
    total = 0.0
    for joint, weight in JOINT_WEIGHTS.items():
        ideal, tol = VRIKS_REF[joint]
        dev  = abs(m[joint] - ideal)
        frac = 1.0 if dev <= tol else max(0.0, 1.0 - (dev - tol) / (tol * 3.0))
        total += weight * frac
    return int(round(total))

def joint_status(joint, angle):
    """Return ('good'|'warn'|'bad', deviation°)."""
    ideal, tol = VRIKS_REF[joint]
    dev = abs(angle - ideal)
    if dev <= tol:          return "good", dev
    elif dev <= tol * 2.5:  return "warn", dev
    else:                   return "bad",  dev


# ══════════════════════════════════════════════════════════════════════════════
#  BEST METRICS SELECTOR — pick normal or mirror, whichever scores higher
# ══════════════════════════════════════════════════════════════════════════════
def best_metrics_and_score(metrics):
    """
    Compare score for original vs left-right mirrored metrics.
    Returns (best_metrics, best_score, is_mirrored: bool).
    """
    score_normal = compute_score(metrics)
    mm = mirror_metrics(metrics)
    score_mirror = compute_score(mm)

    if score_mirror > score_normal:
        return mm, score_mirror, True
    return metrics, score_normal, False


# ══════════════════════════════════════════════════════════════════════════════
#  FEEDBACK GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_feedback(m):
    fb = {}

    # ── Standing leg (right knee in reference; already resolved to best side) ──
    rk = m["right_knee"]
    if rk >= 170:
        fb["Standing Leg"] = {"status": "good",
            "lines": [f"Standing leg well extended ({rk:.0f}°). Solid foundation!"]}
    elif rk >= 160:
        fb["Standing Leg"] = {"status": "warn",
            "lines": [f"Standing knee slightly bent ({rk:.0f}°). Press into the floor, extend to 175°+.",
                      "Engage your quadriceps and ground through your whole foot."]}
    else:
        fb["Standing Leg"] = {"status": "bad",
            "lines": [f"Standing knee too bent ({rk:.0f}°). Straighten fully — this is your base.",
                      "Distribute weight evenly across your standing foot."]}

    # ── Raised leg (left knee in reference; already resolved to best side) ─────
    lk = m["left_knee"]
    if lk <= 40:
        fb["Raised Leg"] = {"status": "good",
            "lines": [f"Raised knee deeply bent ({lk:.0f}°). Press foot firmly into inner thigh."]}
    elif lk <= 70:
        fb["Raised Leg"] = {"status": "warn",
            "lines": [f"Raised knee partially bent ({lk:.0f}°). Draw foot higher toward inner thigh.",
                      "Aim for the foot to be well above the knee of the standing leg."]}
    else:
        fb["Raised Leg"] = {"status": "bad",
            "lines": [f"Raised leg too straight ({lk:.0f}°). Deeply bend and place foot on inner thigh or calf.",
                      "Never place foot on the inner knee joint — above or below only."]}

    # ── Hips ─────────────────────────────────────────────────────────────────
    ht = m["hip_tilt_deg"]
    if ht <= 5:
        fb["Hip & Pelvis"] = {"status": "good",
            "lines": [f"Hips level ({ht:.1f}%). Excellent pelvis control!"]}
    elif ht <= 10:
        fb["Hip & Pelvis"] = {"status": "warn",
            "lines": [f"Hips slightly uneven ({ht:.1f}%). Draw the raised-leg hip downward.",
                      "Engage obliques to square the pelvis."]}
    else:
        fb["Hip & Pelvis"] = {"status": "bad",
            "lines": [f"Significant hip tilt ({ht:.1f}%). Square your pelvis to the front.",
                      "Think of two headlights on your hip bones facing straight ahead."]}

    # ── Spine ────────────────────────────────────────────────────────────────
    sp  = m["spine"]
    dev = abs(sp - 180.0)
    if dev <= 10:
        fb["Spine & Core"] = {"status": "good",
            "lines": [f"Spine vertical ({sp:.0f}°). Great core engagement!"]}
    elif dev <= 20:
        fb["Spine & Core"] = {"status": "warn",
            "lines": [f"Slight spine lean ({sp:.0f}°). Imagine a string pulling your crown upward.",
                      "Gently engage your abs without holding your breath."]}
    else:
        fb["Spine & Core"] = {"status": "bad",
            "lines": [f"Significant lean ({sp:.0f}°). Engage abs, lift chest, stand tall.",
                      "Try fixing your gaze (drishti) on a still point — it helps balance."]}

    # ── Arms ─────────────────────────────────────────────────────────────────
    avg_el = (m["left_elbow"] + m["right_elbow"]) / 2
    if avg_el >= 160:
        fb["Arms & Shoulders"] = {"status": "good",
            "lines": [f"Arms well extended overhead ({avg_el:.0f}°). Palms together in Namaste!"]}
    elif avg_el >= 135:
        fb["Arms & Shoulders"] = {"status": "warn",
            "lines": [f"Arms partially raised ({avg_el:.0f}°). Extend fully overhead, reach fingertips up.",
                      "Broaden your collar bones and lift from your shoulder blades."]}
    else:
        fb["Arms & Shoulders"] = {"status": "bad",
            "lines": [f"Arms too low ({avg_el:.0f}°). Raise both arms fully above your head.",
                      "If overhead is difficult, keep palms in prayer position at chest height."]}

    # ── Neck & Gaze ──────────────────────────────────────────────────────────
    na = m["neck_deg"]
    if na <= 12:
        fb["Neck & Gaze"] = {"status": "good",
            "lines": [f"Head aligned ({na:.1f}°). Drishti fixed on a still point — excellent focus."]}
    elif na <= 22:
        fb["Neck & Gaze"] = {"status": "warn",
            "lines": [f"Head slightly forward ({na:.1f}°). Draw chin back, ears over shoulders."]}
    else:
        fb["Neck & Gaze"] = {"status": "bad",
            "lines": [f"Head misaligned ({na:.1f}°). Lengthen neck, find a drishti at eye level.",
                      "Avoid looking down — it disrupts balance."]}

    return fb


def get_live_tips(fb):
    tips = []
    for data in fb.values():
        if data["status"] in ("bad", "warn") and data["lines"]:
            tips.append(data["lines"][0])
        if len(tips) >= 2:
            break
    return tips


# ══════════════════════════════════════════════════════════════════════════════
#  SMOOTHERS
# ══════════════════════════════════════════════════════════════════════════════
class Smoother:
    def __init__(self, window=20):
        self.w, self.q = window, []
    def update(self, v):
        self.q.append(v)
        if len(self.q) > self.w: self.q.pop(0)
    def value(self):
        return (int(np.mean(self.q)) if isinstance(self.q[0], (int, np.integer))
                else float(np.mean(self.q))) if self.q else 0


# ══════════════════════════════════════════════════════════════════════════════
#  DRAW HELPERS
# ══════════════════════════════════════════════════════════════════════════════
STATUS_ICON = {"good": "[+]", "warn": "[~]", "bad": "[-]"}
STATUS_COL  = {"good": C["green"], "warn": C["yellow"], "bad": C["red"]}

def draw_waiting(frame, missing):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 120), C["dark"], -1)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, "VRIKSHASANA ANALYZER", (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, C["accent"], 2)
    cv2.putText(frame, "Stand so your FULL BODY is clearly visible in frame.",
                (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.50, C["yellow"], 1)
    msg = f"Landmarks missing: {missing}  |  Waiting for complete body view..."
    cv2.putText(frame, msg, (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, C["orange"], 1)
    cv2.putText(frame, "SPACE: record  |  Q: quit",
                (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["gray"], 1)
    return frame


def draw_overlay(frame, metrics, score, conf, fb, recording, elapsed, live_tips,
                 is_mirrored=False):
    h, w = frame.shape[:2]

    # Top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 100), C["dark"], -1)
    cv2.addWeighted(bar, 0.80, frame, 0.20, 0, frame)

    conf_pct = int(conf * 100)
    conf_col = C["green"] if conf_pct >= 70 else C["yellow"] if conf_pct >= 45 else C["red"]

    # Show mirror indicator in title if pose is mirrored
    mirror_tag = "  [MIRROR]" if is_mirrored else ""
    cv2.putText(frame, f"VRIKSHASANA (TREE POSE) ANALYZER{mirror_tag}", (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, C["accent"], 2)
    cv2.putText(frame, f"Pose match: {conf_pct}%   |  All landmarks detected",
                (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.47, conf_col, 1)
    cv2.putText(frame, "Reference: left_knee=27°  right_knee=175°  spine=180°",
                (15, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.34, C["gray"], 1)

    # Score pill
    sc = C["green"] if score >= 75 else C["yellow"] if score >= 50 else C["red"]
    cv2.rectangle(frame, (w - 148, 8), (w - 8, 92), (28, 28, 28), -1)
    cv2.rectangle(frame, (w - 148, 8), (w - 8, 92), sc, 2)
    cv2.putText(frame, str(score), (w - 130, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.65, sc, 3)
    cv2.putText(frame, "/100", (w - 62, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, C["white"], 1)

    # REC
    if recording:
        cv2.circle(frame, (w - 22, 112), 9, (0, 0, 220), -1)
        cv2.putText(frame, f"REC {elapsed:.0f}s", (w - 130, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 220), 1)

    # Live tips panel
    if live_tips:
        ty = 108
        tp = frame.copy()
        ph = 30 + len(live_tips) * 28
        cv2.rectangle(tp, (0, ty), (w, ty + ph), (10, 10, 45), -1)
        cv2.addWeighted(tp, 0.72, frame, 0.28, 0, frame)
        cv2.putText(frame, "Live corrections:", (10, ty + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C["orange"], 1)
        for i, tip in enumerate(live_tips):
            short = tip[:84] + "..." if len(tip) > 84 else tip
            cv2.putText(frame, f"  > {short}", (10, ty + 18 + (i + 1) * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, C["yellow"], 1)

    # Joint panel (bottom-left)
    joint_rows = [
        ("L knee (raised)",   "left_knee"),
        ("R knee (standing)", "right_knee"),
        ("L hip",             "left_hip"),
        ("R hip",             "right_hip"),
        ("L elbow",           "left_elbow"),
        ("R elbow",           "right_elbow"),
        ("Spine",             "spine"),
    ]
    py = h - len(joint_rows) * 26 - 22
    pv = frame.copy()
    cv2.rectangle(pv, (0, py - 6), (345, h - 16), C["dark"], -1)
    cv2.addWeighted(pv, 0.72, frame, 0.28, 0, frame)

    for i, (label, key) in enumerate(joint_rows):
        y   = py + i * 26
        ang = metrics.get(key, 0.0)
        st, dev = joint_status(key, ang)
        col  = STATUS_COL[st]
        icon = STATUS_ICON[st]
        cv2.putText(frame, f"{icon} {label}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, C["white"], 1)
        cv2.putText(frame,
                    f"{ang:6.1f}°  ref:{VRIKS_REF[key][0]:.0f}°  dev:{dev:.1f}°",
                    (175, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

    cv2.putText(frame, "SPACE: record  |  Q: analyse & quit",
                (15, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["gray"], 1)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════════════════════
def build_report(feedback, avg_metrics, score, is_mirrored=False):
    lines = ["=" * 65,
             "  VRIKSHASANA (TREE POSE) — SESSION ANALYSIS",
             f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             "=" * 65,
             f"\n  OVERALL SCORE: {score} / 100\n"]

    if is_mirrored:
        lines.append("  NOTE: Mirror pose detected (right leg raised). Analysis")
        lines.append("        is fully valid — both orientations score equally.\n")

    if score >= 80:   lines.append("  Excellent! Strong alignment throughout the pose.")
    elif score >= 60: lines.append("  Good effort — targeted corrections will elevate your practice.")
    else:             lines.append("  Keep practising! Focus on the top corrections below.")

    lines += ["\n" + "-" * 65, "  DETAILED FEEDBACK", "-" * 65]
    icons = {"good": "✅", "warn": "⚠️ ", "bad": "❌"}
    for cat, data in feedback.items():
        lines.append(f"\n{icons.get(data['status'], '•')} {cat}")
        for ln in data["lines"]:
            lines.append(f"   • {ln}")

    bad      = [c for c, d in feedback.items() if d["status"] == "bad"]
    warn     = [c for c, d in feedback.items() if d["status"] == "warn"]
    priority = (bad + warn)[:3]

    lines += ["\n" + "-" * 65, "  TOP PRIORITY CORRECTIONS", "-" * 65]
    if not priority:
        lines.append("  No major corrections needed — great work!")
    else:
        for i, cat in enumerate(priority, 1):
            lines.append(f"  {i}. [{cat}] {feedback[cat]['lines'][0]}")

    lines += ["\n" + "-" * 65,
              "  JOINT ANGLES  (session average vs reference)",
              "-" * 65]
    for joint, (ideal, tol) in VRIKS_REF.items():
        measured = avg_metrics.get(joint, 0.0)
        dev      = abs(measured - ideal)
        flag     = "✅" if dev <= tol else "⚠️ " if dev <= tol * 2.5 else "❌"
        lines.append(
            f"  {flag} {joint:<18}: measured {measured:6.1f}°  |"
            f"  ref {ideal:6.1f}°  |  dev {dev:5.1f}°"
        )

    lines.append("\n" + "=" * 65)
    return "\n".join(lines)


def save_outputs(report_text, frames, out_dir="yoga_reports"):
    Path(out_dir).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rp = Path(out_dir) / f"vrikshasana_report_{ts}.txt"
    rp.write_text(report_text, encoding="utf-8")
    sp = None
    if frames:
        sp = Path(out_dir) / f"vrikshasana_snapshot_{ts}.jpg"
        cv2.imwrite(str(sp), frames[len(frames) // 2])
    return str(rp), str(sp) if sp else None


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Camera not found. Try VideoCapture(1).")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    score_sm = Smoother(window=20)
    conf_sm  = Smoother(window=25)

    recording        = False
    recorded_frames  = []
    metrics_history  = []   # stores best_metrics each frame
    mirrored_history = []   # stores is_mirrored flag each frame
    start_time       = 0
    frame_count      = 0
    live_tips        = []
    score            = 0
    conf             = 0.0
    fb               = {}
    is_mirrored      = False

    print("\n╔══════════════════════════════════════════════════╗")
    print("║   🧘  Vrikshasana (Tree Pose) Analyzer           ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  Stand so ALL body parts are visible in frame.  ║")
    print("║  Both left-leg and right-leg raised are valid!  ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  SPACE → Start / Stop recording                  ║")
    print("║  Q     → Analyse session & quit                  ║")
    print("╚══════════════════════════════════════════════════╝\n")

    with mp_pose.Pose(
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
        model_complexity=1,
    ) as pose_model:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            h, w    = frame.shape[:2]
            results = pose_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            raw_metrics = None

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark

                # Always draw skeleton
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                all_vis, missing = all_landmarks_visible(lms)

                if not all_vis:
                    # ── Gate: show prompt until all landmarks appear ──────────
                    frame = draw_waiting(frame, missing)
                else:
                    # ── Full analysis ─────────────────────────────────────────
                    try:
                        raw_metrics = extract_metrics(lms, w, h)

                        raw_conf = detect_vrikshasana(raw_metrics)
                        conf_sm.update(raw_conf)
                        conf = conf_sm.value()

                        # ── Mirror-pose check ─────────────────────────────────
                        best_m, raw_score, is_mirrored = best_metrics_and_score(raw_metrics)

                        score_sm.update(raw_score)
                        score = score_sm.value()

                        fb = generate_feedback(best_m)
                        if frame_count % 30 == 0:
                            live_tips = get_live_tips(fb)

                        frame = draw_overlay(
                            frame, best_m, score, conf, fb,
                            recording,
                            time.time() - start_time if recording else 0,
                            live_tips,
                            is_mirrored=is_mirrored,
                        )
                    except Exception as e:
                        cv2.putText(frame, f"Error: {e}", (15, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C["red"], 1)
            else:
                # No pose detected at all
                frame = draw_waiting(frame, 33)

            if recording and raw_metrics is not None:
                # Store best_m (already resolved above) and mirror flag
                recorded_frames.append(frame.copy())
                best_m_stored, _, mir_stored = best_metrics_and_score(raw_metrics)
                metrics_history.append(best_m_stored.copy())
                mirrored_history.append(mir_stored)

            frame_count += 1
            cv2.imshow("Vrikshasana Analyzer", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if not recording:
                    recording  = True
                    start_time = time.time()
                    recorded_frames.clear()
                    metrics_history.clear()
                    mirrored_history.clear()
                    print("  ● Recording started...")
                else:
                    recording = False
                    print(f"  ■ Stopped — {len(recorded_frames)} frames captured.")
            elif key in (ord("q"), ord("Q"), 27):
                recording = False
                break

    cap.release()
    cv2.destroyAllWindows()

    # ── Post-session analysis ─────────────────────────────────────────────────
    if len(metrics_history) < 5:
        print("\n  Too few frames — record for at least 3 seconds next time.")
        return

    print(f"\n⚙️   Analysing {len(metrics_history)} frames...")

    # Average metrics over the session (already mirror-resolved per frame)
    avg = {
        k: float(np.mean([m[k] for m in metrics_history if k in m]))
        for k in metrics_history[0]
    }

    # Determine dominant orientation for the report
    mirror_fraction = sum(mirrored_history) / len(mirrored_history)
    session_mirrored = mirror_fraction > 0.5

    final_score = compute_score(avg)
    feedback    = generate_feedback(avg)
    report      = build_report(feedback, avg, final_score,
                               is_mirrored=session_mirrored)

    print("\n" + report)
    rp, sp = save_outputs(report, recorded_frames)
    print(f"\n✅  Report saved  → {rp}")
    if sp:
        print(f"📷  Snapshot saved → {sp}")


if __name__ == "__main__":
    run()
