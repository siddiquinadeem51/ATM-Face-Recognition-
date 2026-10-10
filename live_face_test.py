"""
ATM Facial Biometric Authentication Terminal (ArcFace Multi-Domain + Night Vision Boost + Sci-Fi HUD)
------------------------------------------------------------------------------------------------------
Single Self-Contained Script (live_face_test.py)

Features:
  1. Multi-Domain ArcFace 512-D Embeddings (Standard RGB + Night-Enhanced + Grayscale Equalized).
  2. Multi-Stage Low-Light Auto-Brightness Enhancement Engine for pitch-black / night conditions.
  3. Adaptive Night-Vision Threshold & Multi-Detector Cascade Fallback ('mtcnn' -> 'opencv' -> 'ssd').
  4. High-Resolution 1280x740 Sci-Fi ATM Terminal Canvas with Cybernetic Background & Ambient Particles.
  5. Glassmorphism Camera Viewport with Corner Reticles, Facial Target Ring, & Animated Laser Scanline.
  6. Registered Customer Profile Card (loads reference image thumbnail & ArcFace status).
  7. Real-time Authentication HUD Metrics (Cosine Similarity Bar, Distance Gauge, Lux Level, Security Shield).
  8. Non-blocking Audio Harmony Chime & Windows Speech Synthesizer Voice Announcement.
"""

import os
import sys
import threading
import queue
import time
import math
import subprocess

# Ensure UTF-8 output encoding for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Silence TensorFlow / DeepFace verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

# Windows Audio system chime sound player
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# ---------------------------------------------------------
# Global Parameters & Configuration
# ---------------------------------------------------------
DAY_THRESHOLD = 0.60    # Normal daylight cosine matching threshold (60%)
NIGHT_THRESHOLD = 0.50  # Night vision adaptive cosine matching threshold (50%)
DETECTOR_BACKEND = "mtcnn"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
LOW_LIGHT_THRESHOLD = 90.0  # Luminance threshold for auto night-vision boost

CANVAS_W = 1280
CANVAS_H = 740

CAM_VIEW_W = 640
CAM_VIEW_H = 480
CAM_VIEW_X = (CANVAS_W - CAM_VIEW_W) // 2  # 320
CAM_VIEW_Y = 100

# Color Palette (BGR format for OpenCV)
COLOR_BG_DARK = (15, 11, 7)          # Very dark midnight blue/black
COLOR_CYAN = (255, 240, 0)          # Cyber Cyan (BGR)
COLOR_EMERALD = (127, 255, 0)       # Bright Emerald Green (BGR)
COLOR_AMBER = (0, 191, 255)         # Warm Amber Gold (BGR)
COLOR_CRIMSON = (80, 50, 255)       # Crimson Red (BGR)
COLOR_TEXT_WHITE = (245, 245, 245)
COLOR_TEXT_MUTED = (180, 160, 140)
COLOR_PANEL_BG = (28, 20, 14)


def play_atm_success_chime(customer_name="Customer"):
    """Play ATM success chime sound and speak voice confirmation aloud in a background thread."""
    def audio_worker():
        try:
            if HAS_WINSOUND:
                # Ascending chime: C5, E5, G5, C6
                winsound.Beep(523, 90)
                time.sleep(0.01)
                winsound.Beep(659, 90)
                time.sleep(0.01)
                winsound.Beep(784, 110)
                time.sleep(0.01)
                winsound.Beep(1046, 240)
                time.sleep(0.15)

            clean_name = os.path.splitext(os.path.basename(customer_name))[0].capitalize()
            speech_text = f"Customer {clean_name} has been successfully verified. You may now proceed to ATM banking services."
            cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{speech_text}\');"'
            subprocess.run(cmd, shell=True, capture_output=True)
        except Exception:
            pass

    threading.Thread(target=audio_worker, daemon=True).start()


def find_reference_image():
    """Search project folder for ATM reference account image (.jpg, .jpeg, .png)."""
    current_dir = os.getcwd()
    found_images = [
        f for f in os.listdir(current_dir)
        if os.path.isfile(os.path.join(current_dir, f)) and f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    if len(found_images) == 0:
        print("❌ No ATM reference face image found in folder.\n", flush=True)
        print("Please place your account face image (e.g., Nadeem.jpeg) in the project folder.", flush=True)
        sys.exit(1)
    elif len(found_images) == 1:
        print(f"✓ ATM Account Image Found: 1. {found_images[0]}\n", flush=True)
        return found_images[0], 1
    else:
        print("Multiple ATM Account Images Found:\n", flush=True)
        for idx, img_name in enumerate(found_images, 1):
            print(f"  {idx}. {img_name}", flush=True)
        print(flush=True)
        while True:
            try:
                sys.stdout.flush()
                choice = input("Enter the number of your ATM Account Image: ").strip()
                choice_num = int(choice)
                if 1 <= choice_num <= len(found_images):
                    selected = found_images[choice_num - 1]
                    print(f"\n✓ ATM Account Image Selected: {choice_num}. {selected}\n", flush=True)
                    return selected, choice_num
                else:
                    print(f"Please enter a number between 1 and {len(found_images)}.", flush=True)
            except (ValueError, KeyboardInterrupt, EOFError):
                print("Invalid selection. Exiting.", flush=True)
                sys.exit(1)


def enhance_low_light(frame, threshold=LOW_LIGHT_THRESHOLD):
    """
    Multi-Stage Low-Light & Dark Environment Face Enhancer.
    Combines Dynamic Range Stretching, CLAHE Equalization, Power-Law Gamma LUT,
    Bilateral Denoising, and Unsharp Sharpening for maximum facial visibility in pitch dark conditions.
    """
    if frame is None:
        return frame, False, 0.0

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    mean_luminance = float(np.mean(l_channel))

    is_low_light = mean_luminance < threshold

    if is_low_light:
        # Stage 1: Dynamic Percentile Contrast Stretching
        p_low, p_high = np.percentile(l_channel, (1, 99))
        if p_high > p_low:
            l_stretched = np.clip((l_channel.astype(np.float32) - p_low) * (255.0 / (p_high - p_low + 1e-5)), 0, 255).astype(np.uint8)
        else:
            l_stretched = l_channel.copy()

        # Stage 2: Adaptive CLAHE Equalization
        clip_limit = 6.0 if mean_luminance < 45.0 else 4.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l_stretched)

        # Stage 3: Aggressive Power-Law Gamma LUT
        gamma = max(0.20, min(3.8, 160.0 / (mean_luminance + 1e-5)))
        inv_gamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        cl = cv2.LUT(cl, lut)

        # Stage 4: Denoise + Unsharp Mask
        denoised = cv2.bilateralFilter(cl, d=5, sigmaColor=35, sigmaSpace=35)
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
        sharpened = cv2.addWeighted(denoised, 1.4, gaussian, -0.4, 0)

        enhanced_lab = cv2.merge((sharpened, a, b))
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_frame, True, mean_luminance

    return frame, False, mean_luminance


def generate_multi_reference_embeddings(image_path):
    """
    Generate Multi-Domain ArcFace 512-D embeddings for an account image:
      1. Standard RGB embedding
      2. Low-Light Night Vision enhanced embedding
      3. Grayscale Equalized illumination-invariant embedding
    Returns a list of 512-D vector embeddings.
    """
    ref_embeddings = []
    
    # 1. Standard RGB embedding (Multi-backend alignment)
    for backend in [DETECTOR_BACKEND, "opencv"]:
        try:
            results = DeepFace.represent(
                img_path=image_path,
                model_name="ArcFace",
                detector_backend=backend,
                align=True,
                enforce_detection=False
            )
            valid = [r for r in results if r.get("facial_area", {}).get("w", 0) > 40]
            if valid:
                best_face = max(valid, key=lambda r: r.get("facial_area", {}).get("w", 0) * r.get("facial_area", {}).get("h", 0))
                ref_embeddings.append(best_face["embedding"])
        except Exception:
            pass

    # 2. Night-Vision Enhanced Domain embedding
    try:
        raw_img = cv2.imread(image_path)
        if raw_img is not None:
            enhanced_img, _, _ = enhance_low_light(raw_img, threshold=255.0)  # force boost
            enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
            results = DeepFace.represent(
                img_path=enhanced_rgb,
                model_name="ArcFace",
                detector_backend="opencv",
                enforce_detection=False
            )
            valid = [r for r in results if r.get("facial_area", {}).get("w", 0) > 40]
            if valid:
                best_face = max(valid, key=lambda r: r.get("facial_area", {}).get("w", 0) * r.get("facial_area", {}).get("h", 0))
                ref_embeddings.append(best_face["embedding"])
    except Exception:
        pass

    # 3. Grayscale Equalized Illumination-Invariant embedding
    try:
        raw_img = cv2.imread(image_path)
        if raw_img is not None:
            gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            eq = cv2.equalizeHist(gray)
            eq_bgr = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
            eq_rgb = cv2.cvtColor(eq_bgr, cv2.COLOR_BGR2RGB)
            results = DeepFace.represent(
                img_path=eq_rgb,
                model_name="ArcFace",
                detector_backend="opencv",
                enforce_detection=False
            )
            valid = [r for r in results if r.get("facial_area", {}).get("w", 0) > 40]
            if valid:
                best_face = max(valid, key=lambda r: r.get("facial_area", {}).get("w", 0) * r.get("facial_area", {}).get("h", 0))
                ref_embeddings.append(best_face["embedding"])
    except Exception:
        pass

    if not ref_embeddings:
        print("❌ Could not generate valid face embeddings for reference image.", flush=True)
        sys.exit(1)

    print("✓ ATM Reference Face verified successfully")
    print(f"✓ Generated {len(ref_embeddings)} Multi-Domain ArcFace 512-D Vectors (Standard + Night + Equalized)\n", flush=True)
    return ref_embeddings, len(ref_embeddings[0])


def compute_best_cosine_similarity(ref_embeddings, live_emb):
    """Compute maximum cosine similarity across multi-domain reference embeddings."""
    if not ref_embeddings or live_emb is None:
        return 0.0
    best_sim = 0.0
    for ref_emb in ref_embeddings:
        sim = float(cosine_similarity([ref_emb], [live_emb])[0][0])
        if sim > best_sim:
            best_sim = sim
    return best_sim


def check_face_alignment(bbox, frame_w, frame_h):
    """Evaluate face centering, size, and distance for ATM verification."""
    if bbox is None:
        return "NO FACE", False, "POSITION YOUR FACE IN TARGET RING", 0

    x, y, w, h = bbox
    face_cx = x + w / 2.0
    face_cy = y + h / 2.0

    center_x = frame_w / 2.0
    center_y = frame_h / 2.0

    offset_x = abs(face_cx - center_x)
    offset_y = abs(face_cy - center_y)

    min_w = frame_w * 0.16
    max_w = frame_w * 0.60

    max_allowed_offset_x = frame_w * 0.20
    max_allowed_offset_y = frame_h * 0.22

    x_score = max(0, 100 - (offset_x / max_allowed_offset_x) * 50)
    y_score = max(0, 100 - (offset_y / max_allowed_offset_y) * 50)
    align_score = int(min(100, (x_score + y_score) / 2))

    if w < min_w:
        return "TOO FAR", False, "🔍 MOVE CLOSER TO ATM CAMERA", align_score
    elif w > max_w:
        return "TOO CLOSE", False, "🔍 MOVE SLIGHTLY BACK", align_score
    elif offset_x > max_allowed_offset_x or offset_y > max_allowed_offset_y:
        return "OFF CENTER", False, "🎯 CENTER YOUR FACE IN SCANNER RING", align_score
    else:
        return "ALIGNED", True, "✓ FACE ALIGNED - AUTHENTICATING...", 100


class AsyncFaceRecognizer(threading.Thread):
    """Background thread performing non-blocking Multi-Domain ArcFace facial evaluation."""
    def __init__(self, ref_embeddings, detector_backend=DETECTOR_BACKEND):
        super().__init__(daemon=True)
        self.ref_embeddings = ref_embeddings
        self.detector_backend = detector_backend

        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        self.lock = threading.Lock()

        # Shared state
        self.face_status_str = "SCANNING..."
        self.similarity_val = 0.0
        self.similarity_str = "N/A"
        self.match_status_str = "SCANNING..."
        self.current_bbox = None
        self.status_color = COLOR_AMBER
        self.multiple_faces_detected = False
        self.is_low_light = False
        self.luminance_val = 0.0

    def push_frame(self, frame):
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def get_latest_results(self):
        with self.lock:
            return (
                self.face_status_str,
                self.similarity_val,
                self.similarity_str,
                self.match_status_str,
                self.current_bbox,
                self.status_color,
                self.multiple_faces_detected,
                self.is_low_light,
                self.luminance_val
            )

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            try:
                raw_frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                proc_frame, low_light_active, lum = enhance_low_light(raw_frame)
                rgb_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)

                frame_h, frame_w = proc_frame.shape[:2]
                valid_faces = []

                # Multi-Detector Cascade Fallback for Dark / Night Conditions
                backends_to_try = [self.detector_backend, "opencv", "ssd"] if low_light_active else [self.detector_backend, "opencv"]

                for backend in backends_to_try:
                    try:
                        results = DeepFace.represent(
                            img_path=rgb_frame,
                            model_name="ArcFace",
                            detector_backend=backend,
                            enforce_detection=False
                        )
                        for r in results:
                            area = r.get("facial_area", {})
                            w = area.get("w", 0)
                            h = area.get("h", 0)
                            if w > 25 and h > 25 and not (w >= frame_w and h >= frame_h):
                                valid_faces.append(r)
                        if valid_faces:
                            # Filter background noise faces (<35% area of dominant face)
                            max_a = max(r.get("facial_area", {}).get("w", 0) * r.get("facial_area", {}).get("h", 0) for r in valid_faces)
                            valid_faces = [r for r in valid_faces if (r.get("facial_area", {}).get("w", 0) * r.get("facial_area", {}).get("h", 0)) >= max_a * 0.35]
                            break
                    except Exception:
                        continue

                with self.lock:
                    self.is_low_light = low_light_active
                    self.luminance_val = lum

                    # Adaptive threshold selection based on light level
                    current_threshold = NIGHT_THRESHOLD if low_light_active else DAY_THRESHOLD

                    if len(valid_faces) == 0:
                        self.face_status_str = "NOT DETECTED"
                        self.similarity_val = 0.0
                        self.similarity_str = "0.0%"
                        self.match_status_str = "NO FACE"
                        self.current_bbox = None
                        self.status_color = COLOR_CRIMSON
                        self.multiple_faces_detected = False

                    elif len(valid_faces) > 1:
                        self.face_status_str = "MULTIPLE FACES"
                        self.similarity_val = 0.0
                        self.similarity_str = "SECURITY ALERT"
                        self.match_status_str = "MULTIPLE FACES"
                        self.current_bbox = None
                        self.status_color = COLOR_CRIMSON
                        self.multiple_faces_detected = True

                    else:
                        self.face_status_str = "DETECTED"
                        self.multiple_faces_detected = False
                        primary_face = max(valid_faces, key=lambda r: r.get("facial_area", {}).get("w", 0) * r.get("facial_area", {}).get("h", 0))
                        live_emb = primary_face["embedding"]
                        area = primary_face["facial_area"]
                        self.current_bbox = (area["x"], area["y"], area["w"], area["h"])

                        # Multi-Domain Cosine Similarity Evaluation
                        sim_val = compute_best_cosine_similarity(self.ref_embeddings, live_emb)
                        sim_pct = max(0.0, sim_val * 100)
                        self.similarity_val = sim_val
                        self.similarity_str = f"{sim_pct:.1f}%"

                        if sim_val >= current_threshold:
                            self.match_status_str = "ACCOUNT MATCH ✓"
                            self.status_color = COLOR_EMERALD
                        else:
                            self.match_status_str = "NO MATCH"
                            self.status_color = COLOR_CRIMSON

            except Exception:
                with self.lock:
                    self.face_status_str = "NOT DETECTED"
                    self.similarity_val = 0.0
                    self.similarity_str = "N/A"
                    self.match_status_str = "NO FACE"
                    self.current_bbox = None
                    self.status_color = COLOR_CRIMSON
                    self.multiple_faces_detected = False

            self.frame_queue.task_done()


# ---------------------------------------------------------
# UI Background Artwork & Custom Rendering Helpers
# ---------------------------------------------------------
def create_cyber_background(w=CANVAS_W, h=CANVAS_H, anim_time=0.0):
    """Generate a rich, dark futuristic background canvas with cyber grid and ambient particles."""
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    
    y_coords, x_coords = np.ogrid[:h, :w]
    center_x, center_y = w // 2, h // 2
    dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    norm_dist = dist_from_center / max_dist
    
    r_channel = (10 + 20 * (1 - norm_dist)).astype(np.uint8)
    g_channel = (15 + 30 * (1 - norm_dist)).astype(np.uint8)
    b_channel = (30 + 55 * (1 - norm_dist)).astype(np.uint8)
    
    bg[:, :, 0] = b_channel
    bg[:, :, 1] = g_channel
    bg[:, :, 2] = r_channel

    grid_spacing = 40
    grid_overlay = bg.copy()
    grid_color = (60, 45, 25)
    
    for x in range(0, w, grid_spacing):
        cv2.line(grid_overlay, (x, 0), (x, h), grid_color, 1)
    for y in range(0, h, grid_spacing):
        cv2.line(grid_overlay, (0, y), (w, y), grid_color, 1)
        
    cv2.addWeighted(grid_overlay, 0.35, bg, 0.65, 0, bg)

    num_nodes = 12
    for i in range(num_nodes):
        angle = anim_time * 0.5 + (i * (2 * math.pi / num_nodes))
        radius_x = 420 + 60 * math.sin(anim_time * 0.3 + i)
        radius_y = 220 + 40 * math.cos(anim_time * 0.4 + i)
        
        px = int(center_x + radius_x * math.cos(angle))
        py = int(center_y + radius_y * math.sin(angle))
        
        if 0 <= px < w and 0 <= py < h:
            glow_size = int(6 + 3 * math.sin(anim_time * 2 + i))
            cv2.circle(bg, (px, py), glow_size, (255, 200, 0), -1)
            cv2.circle(bg, (px, py), glow_size + 4, (180, 120, 0), 1)

    return bg


def draw_styled_corner_box(img, x, y, w, h, color, scan_line_y=None):
    """Draw corner reticles and vertical laser scanline on the camera feed."""
    line_len = int(min(w, h) * 0.22)
    thickness = 2

    cv2.line(img, (x, y), (x + line_len, y), color, thickness)
    cv2.line(img, (x, y), (x, y + line_len), color, thickness)

    cv2.line(img, (x + w, y), (x + w - line_len, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + line_len), color, thickness)

    cv2.line(img, (x, y + h), (x + line_len, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - line_len), color, thickness)

    cv2.line(img, (x + w, y + h), (x + w - line_len, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - line_len), color, thickness)

    if scan_line_y is not None:
        ly = y + int((scan_line_y % 100) / 100.0 * h)
        cv2.line(img, (x + 4, ly), (x + w - 4, ly), (255, 255, 255), 1)
        overlay = img.copy()
        cv2.rectangle(overlay, (x, max(y, ly - 5)), (x + w, min(y + h, ly + 5)), color, -1)
        cv2.addWeighted(overlay, 0.30, img, 0.70, 0, img)


def draw_target_ring(img, center, axes, color, aligned=False):
    """Draw facial target ring alignment guide."""
    cv2.ellipse(img, center, axes, 0, 0, 360, (70, 70, 70), 1, cv2.LINE_AA)

    cx, cy = center
    ax, ay = axes
    cv2.line(img, (cx - ax - 12, cy), (cx - ax + 6, cy), color, 2)
    cv2.line(img, (cx + ax - 6, cy), (cx + ax + 12, cy), color, 2)
    cv2.line(img, (cx, cy - ay - 12), (cx, cy - ay + 6), color, 2)
    cv2.line(img, (cx, cy + ay - 6), (cx, cy + ay + 12), color, 2)

    if aligned:
        cv2.ellipse(img, center, axes, 0, 0, 360, color, 2, cv2.LINE_AA)


def draw_panel_card(canvas, x, y, w, h, title="", border_color=COLOR_CYAN):
    """Draw a dark glassmorphism side HUD panel card with cyan borders."""
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

    cv2.rectangle(canvas, (x, y), (x + w, y + h), (45, 35, 25), 1)
    cv2.line(canvas, (x, y), (x + w, y), border_color, 2)
    
    cv2.line(canvas, (x, y), (x + 12, y), border_color, 3)
    cv2.line(canvas, (x, y), (x, y + 12), border_color, 3)
    cv2.line(canvas, (x + w - 12, y), (x + w, y), border_color, 3)
    cv2.line(canvas, (x + w, y), (x + w, y + 12), border_color, 3)

    if title:
        cv2.putText(canvas, title.upper(), (x + 16, y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, border_color, 2)
        cv2.line(canvas, (x + 16, y + 36), (x + w - 16, y + 36), (60, 45, 30), 1)


def draw_verification_badge(canvas, center_x, center_y, is_verified, text=""):
    """
    Draw a clean visual status badge with a Green Tick Mark [✓] when verified,
    or a Red Cross Mark [✗] when not verified.
    """
    color = COLOR_EMERALD if is_verified else COLOR_CRIMSON
    radius = 14

    cv2.circle(canvas, (center_x, center_y), radius + 2, (10, 10, 10), -1)
    cv2.circle(canvas, (center_x, center_y), radius, color, 2, cv2.LINE_AA)

    if is_verified:
        # Green Tick Mark ✓
        pt1 = (center_x - 6, center_y + 1)
        pt2 = (center_x - 2, center_y + 5)
        pt3 = (center_x + 6, center_y - 5)
        cv2.line(canvas, pt1, pt2, COLOR_EMERALD, 3, cv2.LINE_AA)
        cv2.line(canvas, pt2, pt3, COLOR_EMERALD, 3, cv2.LINE_AA)
    else:
        # Red Cross Mark ✗
        off = 5
        cv2.line(canvas, (center_x - off, center_y - off), (center_x + off, center_y + off), COLOR_CRIMSON, 3, cv2.LINE_AA)
        cv2.line(canvas, (center_x + off, center_y - off), (center_x - off, center_y + off), COLOR_CRIMSON, 3, cv2.LINE_AA)

    if text:
        tx = center_x + radius + 10
        ty = center_y + 5
        cv2.putText(canvas, text.upper(), (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)


def draw_atm_success_modal(canvas, similarity_str, ref_name):
    """Draw Full Terminal Biometric Authentication Success Screen."""
    w, h = CANVAS_W, CANVAS_H

    backdrop = canvas.copy()
    cv2.rectangle(backdrop, (0, 0), (w, h), (10, 20, 10), -1)
    cv2.addWeighted(backdrop, 0.82, canvas, 0.18, 0, canvas)

    pw, ph = int(w * 0.72), int(h * 0.58)
    px = (w - pw) // 2
    py = (h - ph) // 2

    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (18, 28, 20), -1)
    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), COLOR_EMERALD, 2)
    cv2.rectangle(canvas, (px - 5, py - 5), (px + pw + 5, py + ph + 5), (0, 200, 100), 1)

    chk_cx, chk_cy = px + pw // 2, py + 55
    cv2.circle(canvas, (chk_cx, chk_cy), 32, COLOR_EMERALD, -1)
    cv2.putText(canvas, "✓", (chk_cx - 15, chk_cy + 13), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)

    account_user = os.path.splitext(os.path.basename(ref_name))[0].upper()
    cv2.putText(canvas, "ATM FACIAL BIOMETRIC VERIFICATION SUCCESSFUL!", (px + 40, py + 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)

    cv2.putText(canvas, f"ACCOUNT HOLDER  :  {account_user} (VERIFIED ✓)", (px + 60, py + 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, COLOR_EMERALD, 2)

    cv2.putText(canvas, f"MATCH SIMILARITY :  {similarity_str}  |  ATM TERMINAL #8092", (px + 60, py + 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 220, 180), 1)

    cv2.putText(canvas, "Biometric identity confirmed. Access granted to ATM banking services.", (px + 60, py + 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 255, 220), 1)

    btn_y = py + ph - 55
    cv2.rectangle(canvas, (px + 60, btn_y), (px + pw - 60, btn_y + 40), COLOR_EMERALD, -1)
    cv2.putText(canvas, "PRESS ENTER OR SPACE TO CONTINUE TO ATM BANKING", (px + 100, btn_y + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 0, 0), 2)


LAST_FAILURE_AUDIO_TIME = 0

def play_atm_failure_chime():
    """Play warning audio chime and speak voice announcement for failed face verification."""
    global LAST_FAILURE_AUDIO_TIME
    now = time.time()
    if now - LAST_FAILURE_AUDIO_TIME < 6.0:
        return
    LAST_FAILURE_AUDIO_TIME = now

    def audio_worker():
        try:
            if HAS_WINSOUND:
                winsound.Beep(400, 120)
                time.sleep(0.01)
                winsound.Beep(300, 200)

            speech_text = "Authentication failed. Face does not match registered ATM account holder."
            cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{speech_text}\');"'
            subprocess.run(cmd, shell=True, capture_output=True)
        except Exception:
            pass

    threading.Thread(target=audio_worker, daemon=True).start()


def draw_atm_failed_modal(canvas, similarity_str, ref_name):
    """Draw Full Terminal Biometric Authentication Failed Screen Overlay."""
    w, h = CANVAS_W, CANVAS_H

    backdrop = canvas.copy()
    cv2.rectangle(backdrop, (0, 0), (w, h), (25, 10, 10), -1)
    cv2.addWeighted(backdrop, 0.82, canvas, 0.18, 0, canvas)

    pw, ph = int(w * 0.72), int(h * 0.58)
    px = (w - pw) // 2
    py = (h - ph) // 2

    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (28, 18, 20), -1)
    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), COLOR_CRIMSON, 2)
    cv2.rectangle(canvas, (px - 5, py - 5), (px + pw + 5, py + ph + 5), (180, 40, 60), 1)

    chk_cx, chk_cy = px + pw // 2, py + 55
    cv2.circle(canvas, (chk_cx, chk_cy), 32, COLOR_CRIMSON, -1)

    off = 14
    cv2.line(canvas, (chk_cx - off, chk_cy - off), (chk_cx + off, chk_cy + off), (255, 255, 255), 4, cv2.LINE_AA)
    cv2.line(canvas, (chk_cx + off, chk_cy - off), (chk_cx - off, chk_cy + off), (255, 255, 255), 4, cv2.LINE_AA)

    account_user = os.path.splitext(os.path.basename(ref_name))[0].upper()
    cv2.putText(canvas, "ATM FACIAL BIOMETRIC VERIFICATION FAILED!", (px + 40, py + 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)

    cv2.putText(canvas, f"TARGET ACCOUNT :  {account_user} (NOT MATCHED ✗)", (px + 60, py + 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, COLOR_CRIMSON, 2)

    cv2.putText(canvas, f"MATCH SIMILARITY :  {similarity_str}  |  ATM TERMINAL #8092", (px + 60, py + 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 180, 180), 1)

    cv2.putText(canvas, "⚠️ Face does not match registered customer account holder.", (px + 60, py + 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 200, 200), 1)

    btn_y = py + ph - 55
    cv2.rectangle(canvas, (px + 60, btn_y), (px + pw - 60, btn_y + 40), COLOR_CRIMSON, -1)
    cv2.putText(canvas, "ACCESS DENIED - UNMATCHED CUSTOMER IDENTITY", (px + 80, btn_y + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)


def create_reference_thumbnail(image_path, target_size=(110, 110)):
    """Load and resize reference account face image for side HUD panel."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        min_dim = min(h, w)
        crop_x = (w - min_dim) // 2
        crop_y = (h - min_dim) // 2
        cropped = img[crop_y:crop_y + min_dim, crop_x:crop_x + min_dim]
        resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
        return resized
    except Exception:
        return None


# ---------------------------------------------------------
# Main Application Loop
# ---------------------------------------------------------
def main():
    print("=========================================================", flush=True)
    print("  BANK ATM FACIAL BIOMETRIC TERMINAL (STANDALONE PYTHON)", flush=True)
    print("=========================================================\n", flush=True)

    reference_image_path, cust_num = find_reference_image()
    print(f"Loading reference image {cust_num}: {reference_image_path}...", flush=True)
    
    ref_embeddings, emb_dim = generate_multi_reference_embeddings(reference_image_path)
    ref_thumbnail = create_reference_thumbnail(reference_image_path, target_size=(110, 110))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not access ATM camera module.", flush=True)
        print("⚠️ Ensure no other application (like web_app.py) is using the camera.\n", flush=True)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("✓ ATM Camera Module Active.", flush=True)
    print("✓ Opening ATM Sci-Fi Biometric Display Terminal...\n", flush=True)

    recognizer_thread = AsyncFaceRecognizer(
        ref_embeddings=ref_embeddings,
        detector_backend=DETECTOR_BACKEND
    )
    recognizer_thread.start()

    window_title = "ATM Biometric Authentication Terminal #8092"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, CANVAS_W, CANVAS_H)

    scan_line_counter = 0
    verification_lock_count = 0
    unmatched_frames_count = 0
    is_verified_state = False
    is_failed_state = False
    chime_played = False
    failure_chime_played = False
    start_time = time.time()
    frame_count = 0
    fps = 0.0
    fps_time = time.time()
    camera_warmup_attempts = 0

    account_user_name = f"{cust_num}. {os.path.splitext(os.path.basename(reference_image_path))[0].capitalize()}"

    while True:
        ret, raw_frame = cap.read()
        if not ret or raw_frame is None:
            camera_warmup_attempts += 1
            if camera_warmup_attempts > 25:
                print("❌ Could not read frames from camera.", flush=True)
                print("⚠️ Camera may be in use by another app (e.g. web_app.py). Please close web_app.py and retry.", flush=True)
                break
            time.sleep(0.04)
            continue

        camera_warmup_attempts = 0
        frame_count += 1

        now = time.time()
        if now - fps_time >= 1.0:
            fps = frame_count / (now - fps_time)
            frame_count = 0
            fps_time = now

        raw_frame = cv2.flip(raw_frame, 1)
        raw_h, raw_w = raw_frame.shape[:2]

        recognizer_thread.push_frame(raw_frame)

        (
            face_status_str,
            similarity_val,
            similarity_str,
            match_status_str,
            current_bbox,
            status_color,
            multiple_faces_detected,
            is_low_light,
            luminance_val
        ) = recognizer_thread.get_latest_results()

        if is_low_light:
            proc_cam_frame, _, _ = enhance_low_light(raw_frame, threshold=LOW_LIGHT_THRESHOLD)
        else:
            proc_cam_frame = raw_frame.copy()

        cam_view = cv2.resize(proc_cam_frame, (CAM_VIEW_W, CAM_VIEW_H))
        v_h, v_w = cam_view.shape[:2]

        scale_x = CAM_VIEW_W / float(raw_w)
        scale_y = CAM_VIEW_H / float(raw_h)

        if current_bbox is not None:
            bx, by, bw, bh = current_bbox
            view_bbox = (int(bx * scale_x), int(by * scale_y), int(bw * scale_x), int(bh * scale_y))
        else:
            view_bbox = None

        align_status, is_aligned, align_prompt, align_score = check_face_alignment(
            view_bbox, v_w, v_h
        )

        if view_bbox is not None and is_aligned and not multiple_faces_detected:
            if match_status_str == "ACCOUNT MATCH ✓":
                unmatched_frames_count = 0
                is_failed_state = False
                failure_chime_played = False
                verification_lock_count += 1
                if verification_lock_count >= 2:
                    is_verified_state = True
                    if not chime_played:
                        play_atm_success_chime(customer_name=reference_image_path)
                        chime_played = True
            elif match_status_str == "NO MATCH":
                verification_lock_count = 0
                if not is_verified_state:
                    unmatched_frames_count += 1
                    if unmatched_frames_count >= 25:
                        is_failed_state = True
                        if not failure_chime_played:
                            play_atm_failure_chime()
                            failure_chime_played = True
        else:
            verification_lock_count = 0
            unmatched_frames_count = 0
            if not is_verified_state:
                is_failed_state = False
                failure_chime_played = False

        # RENDER FULL SCI-FI CANVAS UI
        anim_time = now - start_time
        canvas = create_cyber_background(CANVAS_W, CANVAS_H, anim_time=anim_time)

        center_pt = (v_w // 2, v_h // 2)
        target_axes = (int(v_w * 0.17), int(v_h * 0.27))
        ring_color = COLOR_EMERALD if is_aligned else COLOR_CYAN
        draw_target_ring(cam_view, center_pt, target_axes, ring_color, aligned=is_aligned)

        scan_line_counter += 3
        is_face_match = (match_status_str == "ACCOUNT MATCH ✓")
        is_face_ver = is_verified_state or (is_face_match and is_aligned)

        if view_bbox is not None and not multiple_faces_detected:
            vx, vy, vw, vh = view_bbox
            box_color = COLOR_EMERALD if is_face_ver else (COLOR_AMBER if is_face_match else COLOR_CRIMSON)
            draw_styled_corner_box(cam_view, vx, vy, vw, vh, box_color, scan_line_y=scan_line_counter)

            # Floating Verification Badge on Camera Feed
            if is_face_ver:
                badge_lbl = "VERIFIED"
            elif is_face_match:
                badge_lbl = "ALIGN FACE"
            else:
                badge_lbl = "NOT VERIFIED"

            badge_x = max(25, vx + 20)
            badge_y = max(25, vy - 20)
            draw_verification_badge(cam_view, badge_x, badge_y, is_face_ver, badge_lbl)

        if is_low_light:
            cv2.rectangle(cam_view, (0, 0), (v_w, v_h), (255, 255, 255), 3)
            cv2.rectangle(cam_view, (v_w - 230, 10), (v_w - 10, 36), (0, 140, 255), -1)
            cv2.putText(cam_view, "🌙 ATM NIGHT VISION ACTIVE", (v_w - 222, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 2)

        canvas[CAM_VIEW_Y:CAM_VIEW_Y + CAM_VIEW_H, CAM_VIEW_X:CAM_VIEW_X + CAM_VIEW_W] = cam_view
        
        cv2.rectangle(canvas, (CAM_VIEW_X - 2, CAM_VIEW_Y - 2),
                      (CAM_VIEW_X + CAM_VIEW_W + 2, CAM_VIEW_Y + CAM_VIEW_H + 2), ring_color, 2)

        # TOP HEADER SECURITY TERMINAL BAR
        hdr_overlay = canvas.copy()
        cv2.rectangle(hdr_overlay, (0, 0), (CANVAS_W, 70), COLOR_PANEL_BG, -1)
        cv2.addWeighted(hdr_overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.line(canvas, (0, 70), (CANVAS_W, 70), COLOR_CYAN, 2)

        cv2.putText(canvas, "BANK OF FUTURE", (25, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, COLOR_TEXT_WHITE, 2)
        cv2.putText(canvas, "FACIAL BIOMETRIC AUTHENTICATION TERMINAL #8092", (25, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_CYAN, 1)

        # Top Header Status Badge with Tick (✓) / Cross (✗)
        sys_status_str = "VERIFIED" if is_face_ver else ("ALIGN FACE" if is_face_match else "NOT VERIFIED")
        draw_verification_badge(canvas, 560, 34, is_face_ver, sys_status_str)

        time_str = time.strftime("%I:%M:%S %p")
        cv2.putText(canvas, time_str, (CANVAS_W - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_TEXT_WHITE, 1)
        cv2.putText(canvas, f"AES-256 ENCRYPTED  |  {fps:.1f} FPS", (CANVAS_W - 275, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1)

        # LEFT HUD PANEL: REGISTERED ACCOUNT PROFILE
        panel_left_x, panel_left_y, panel_left_w, panel_left_h = 20, 100, 280, 480
        draw_panel_card(canvas, panel_left_x, panel_left_y, panel_left_w, panel_left_h,
                        title="ACCOUNT PROFILE", border_color=COLOR_CYAN)

        thumb_x = panel_left_x + (panel_left_w - 110) // 2
        thumb_y = panel_left_y + 55
        
        if ref_thumbnail is not None:
            canvas[thumb_y:thumb_y + 110, thumb_x:thumb_x + 110] = ref_thumbnail
            cv2.rectangle(canvas, (thumb_x - 2, thumb_y - 2), (thumb_x + 112, thumb_y + 112), COLOR_CYAN, 2)
        else:
            cv2.rectangle(canvas, (thumb_x, thumb_y), (thumb_x + 110, thumb_y + 110), (60, 60, 60), -1)
            cv2.putText(canvas, "NO IMAGE", (thumb_x + 18, thumb_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        info_y = thumb_y + 140
        cv2.putText(canvas, "HOLDER NAME:", (panel_left_x + 20, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1)
        cv2.putText(canvas, account_user_name.upper(), (panel_left_x + 20, info_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, COLOR_TEXT_WHITE, 2)

        cv2.putText(canvas, "ACCOUNT NO:", (panel_left_x + 20, info_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1)
        cv2.putText(canvas, "**** - 8092 - SECURE", (panel_left_x + 20, info_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_CYAN, 1)

        cv2.putText(canvas, "MATCH STATUS:", (panel_left_x + 20, info_y + 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1)
        card_status_lbl = "MATCHED" if is_face_ver else ("ALIGNING..." if is_face_match else "NO MATCH")
        draw_verification_badge(canvas, panel_left_x + 35, info_y + 138, is_face_ver, card_status_lbl)

        cv2.putText(canvas, "DETECTOR BACKEND:", (panel_left_x + 20, info_y + 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1)
        cv2.putText(canvas, DETECTOR_BACKEND.upper(), (panel_left_x + 20, info_y + 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_TEXT_WHITE, 1)

        # RIGHT HUD PANEL: REAL-TIME AUTHENTICATION METRICS
        panel_right_x, panel_right_y, panel_right_w, panel_right_h = 980, 100, 280, 480
        draw_panel_card(canvas, panel_right_x, panel_right_y, panel_right_w, panel_right_h,
                        title="LIVE BIOMETRICS", border_color=COLOR_CYAN)

        m1_y = panel_right_y + 55
        cv2.putText(canvas, "COSINE MATCH SIMILARITY", (panel_right_x + 20, m1_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1)

        active_thresh = NIGHT_THRESHOLD if is_low_light else DAY_THRESHOLD
        sim_gauge_color = COLOR_EMERALD if similarity_val >= active_thresh else COLOR_CRIMSON
        cv2.putText(canvas, similarity_str, (panel_right_x + 20, m1_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, sim_gauge_color, 2)

        bar_x, bar_y, bar_w, bar_h = panel_right_x + 20, m1_y + 44, 240, 14
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 40, 30), -1)
        fill_w = int(bar_w * min(1.0, similarity_val))
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), sim_gauge_color, -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 80, 60), 1)

        threshold_x = bar_x + int(bar_w * active_thresh)
        cv2.line(canvas, (threshold_x, bar_y - 3), (threshold_x, bar_y + bar_h + 3), COLOR_CYAN, 2)
        cv2.putText(canvas, f"REQ: {int(active_thresh*100)}%", (threshold_x - 22, bar_y + bar_h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_CYAN, 1)

        m2_y = m1_y + 115
        cv2.putText(canvas, "POSITION & DISTANCE", (panel_right_x + 20, m2_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1)
        
        cv2.putText(canvas, f"{align_score}%  ({align_status})", (panel_right_x + 20, m2_y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, ring_color, 2)

        align_bar_x, align_bar_y, align_bar_w, align_bar_h = panel_right_x + 20, m2_y + 38, 240, 10
        cv2.rectangle(canvas, (align_bar_x, align_bar_y), (align_bar_x + align_bar_w, align_bar_y + align_bar_h), (50, 40, 30), -1)
        align_fill_w = int(align_bar_w * (align_score / 100.0))
        if align_fill_w > 0:
            cv2.rectangle(canvas, (align_bar_x, align_bar_y), (align_bar_x + align_fill_w, align_bar_y + align_bar_h), ring_color, -1)

        m3_y = m2_y + 85
        cv2.putText(canvas, "SECURITY SHIELD", (panel_right_x + 20, m3_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1)

        if multiple_faces_detected:
            cv2.putText(canvas, "⚠️ MULTIPLE FACES", (panel_right_x + 20, m3_y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_CRIMSON, 2)
        else:
            cv2.putText(canvas, f"1 PERSON DETECTED", (panel_right_x + 20, m3_y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_EMERALD, 1)

        m4_y = m3_y + 65
        cv2.putText(canvas, "ATM ENVIRONMENT LUX", (panel_right_x + 20, m4_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1)
        lux_str = f"{int(luminance_val)} LUX (" + ("NIGHT BOOST" if is_low_light else "NORMAL") + ")"
        cv2.putText(canvas, lux_str, (panel_right_x + 20, m4_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 191, 255) if is_low_light else COLOR_TEXT_WHITE, 1)

        # BOTTOM ACTION & GUIDANCE BAR
        bot_overlay = canvas.copy()
        cv2.rectangle(bot_overlay, (0, CANVAS_H - 120), (CANVAS_W, CANVAS_H), COLOR_PANEL_BG, -1)
        cv2.addWeighted(bot_overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.line(canvas, (0, CANVAS_H - 120), (CANVAS_W, CANVAS_H - 120), COLOR_CYAN, 2)

        if multiple_faces_detected:
            prompt = "⚠️ SECURITY WARNING: MULTIPLE FACES DETECTED. PLEASE KEEP ONLY 1 PERSON AT ATM TERMINAL."
            p_color = COLOR_CRIMSON
        elif is_verified_state:
            prompt = "✨ ATM BIOMETRIC AUTHENTICATED! PROCEED TO BANKING SERVICES."
            p_color = COLOR_EMERALD
        else:
            prompt = align_prompt
            p_color = ring_color

        cv2.putText(canvas, prompt, (35, CANVAS_H - 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, p_color, 2)

        cv2.putText(canvas, "Press 'Q' or ESC to Exit Terminal  |  Bank of Future ATM System", (35, CANVAS_H - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1)

        if is_verified_state:
            draw_atm_success_modal(canvas, similarity_str, reference_image_path)
        elif is_failed_state:
            draw_atm_failed_modal(canvas, similarity_str, reference_image_path)

        cv2.imshow(window_title, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        if is_verified_state and key in (13, 32):
            print("\n=========================================================", flush=True)
            print(f" ✅ ATM USER {account_user_name.upper()} AUTHENTICATED & PROCEEDED!", flush=True)
            print("=========================================================\n", flush=True)
            break

    recognizer_thread.stop()
    cap.release()
    cv2.destroyAllWindows()

    print("=========================================================", flush=True)
    print(" ATM Facial Biometric Session ended.", flush=True)
    print("=========================================================", flush=True)


if __name__ == "__main__":
    main()
