"""
ATM Facial Biometric Authentication System (ArcFace + Low Light + Secure Banking HUD)
---------------------------------------------------------------------------------------
Features:
  1. Low-Light Detection & CLAHE + Gamma Enhancement for dark / night ATM environments.
  2. ATM Face Alignment & Distance Position Guidance.
  3. Non-blocking ATM Confirmation Chime Sound on Authentication.
  4. ATM Security Terminal HUD with laser scanline, target oval, and metrics.
  5. ATM Verification Success Banner ("Account Authenticated! Proceed to ATM Services").
"""

import os
import sys
import threading
import queue
import time
import math

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

# Audio player for Windows system sounds / chimes
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# ---------------------------------------------------------
# Global Parameters & Configuration
# ---------------------------------------------------------
THRESHOLD = 0.60  # Matching threshold (60% cosine similarity)
DETECTOR_BACKEND = "mtcnn"  # Options: 'mtcnn', 'opencv', 'retinaface', 'ssd'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
LOW_LIGHT_THRESHOLD = 75.0  # Luminance threshold for auto-enhancement


import subprocess


def play_atm_success_chime(customer_name="Customer"):
    """Play ATM chime sound and speak voice confirmation aloud in a non-blocking background thread."""
    def audio_worker():
        try:
            if HAS_WINSOUND:
                # Ascending harmony chime: C5, E5, G5, C6
                winsound.Beep(523, 90)
                time.sleep(0.01)
                winsound.Beep(659, 90)
                time.sleep(0.01)
                winsound.Beep(784, 110)
                time.sleep(0.01)
                winsound.Beep(1046, 220)
                time.sleep(0.15)

            # Voice announcement sound
            clean_name = os.path.splitext(os.path.basename(customer_name))[0].capitalize()
            speech_text = f"Customer {clean_name} has been successfully verified. You may now proceed to ATM banking services."
            cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{speech_text}\');"'
            subprocess.run(cmd, shell=True, capture_output=True)
        except Exception:
            pass

    threading.Thread(target=audio_worker, daemon=True).start()



def find_reference_image():
    """
    Search project directory for ATM reference account image (.jpg, .jpeg, .png).
    Returns the path to the reference image.
    """
    current_dir = os.getcwd()
    found_images = [
        f for f in os.listdir(current_dir)
        if os.path.isfile(os.path.join(current_dir, f)) and f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    if len(found_images) == 0:
        print("❌ No ATM reference face image found.\n", flush=True)
        print("Please place your account face image in the project folder.", flush=True)
        sys.exit(1)
    elif len(found_images) == 1:
        print(f"ATM Account Image Found:\n -> {found_images[0]}\n", flush=True)
        return found_images[0]
    else:
        print("Multiple ATM Account Images Found:\n", flush=True)
        for idx, img_name in enumerate(found_images, 1):
            print(f"{idx}. {img_name}", flush=True)
        print(flush=True)
        while True:
            try:
                sys.stdout.flush()
                choice = input("Enter the number of your ATM Account Image: ").strip()
                choice_num = int(choice)
                if 1 <= choice_num <= len(found_images):
                    selected = found_images[choice_num - 1]
                    print(f"\nATM Account Image Selected:\n -> {selected}\n", flush=True)
                    return selected
                else:
                    print(f"Please enter a number between 1 and {len(found_images)}.", flush=True)
            except (ValueError, KeyboardInterrupt, EOFError):
                print("Invalid selection. Exiting.", flush=True)
                sys.exit(1)


def enhance_low_light(frame, threshold=LOW_LIGHT_THRESHOLD):
    """
    Detect low-light night ATM environments and apply CLAHE + adaptive gamma correction.
    Returns: (enhanced_frame, is_low_light, mean_luminance)
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    mean_luminance = float(np.mean(l_channel))

    is_low_light = mean_luminance < threshold

    if is_low_light:
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)

        gamma = max(0.4, min(2.5, 130.0 / (mean_luminance + 1e-5)))
        inv_gamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        cl = cv2.LUT(cl, lut)

        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_frame, True, mean_luminance

    return frame, False, mean_luminance


def generate_reference_embedding(image_path, detector_backend=DETECTOR_BACKEND):
    """Validate ATM reference image and generate ArcFace embedding."""
    try:
        results = DeepFace.represent(
            img_path=image_path,
            model_name="ArcFace",
            detector_backend=detector_backend,
            enforce_detection=True
        )
    except Exception:
        if detector_backend != "opencv":
            print(f"⚠️ Detector '{detector_backend}' failed on ATM image, trying 'opencv'...", flush=True)
            return generate_reference_embedding(image_path, detector_backend="opencv")
        print("❌ No face detected in the ATM account reference image.\n", flush=True)
        sys.exit(1)

    valid_results = [
        r for r in results
        if r.get("facial_area", {}).get("w", 0) > 60 and r.get("facial_area", {}).get("h", 0) > 60
    ]

    if not valid_results:
        print("❌ No valid face detected in the ATM reference image.\n", flush=True)
        sys.exit(1)
    if len(valid_results) > 1:
        print("❌ Multiple faces detected in the ATM reference image.\n", flush=True)
        sys.exit(1)

    embedding = valid_results[0]["embedding"]
    embedding_dim = len(embedding)
    print("✓ ATM Reference Face verified successfully", flush=True)
    print(f"✓ ArcFace 512-D Biometric Embedding Generated using {detector_backend}", flush=True)
    return embedding, embedding_dim


def check_face_alignment(bbox, frame_w, frame_h):
    """
    Evaluate face positioning, centering, scale/distance for ATM scan.
    Returns: (alignment_status, is_aligned, guidance_prompt, align_score)
    """
    if bbox is None:
        return "NO FACE", False, "ATM SCAN: POSITION YOUR FACE IN FRAME", 0

    x, y, w, h = bbox
    face_cx = x + w / 2.0
    face_cy = y + h / 2.0

    center_x = frame_w / 2.0
    center_y = frame_h / 2.0

    offset_x = abs(face_cx - center_x)
    offset_y = abs(face_cy - center_y)

    min_w = frame_w * 0.18
    max_w = frame_w * 0.58

    max_allowed_offset_x = frame_w * 0.18
    max_allowed_offset_y = frame_h * 0.20

    x_score = max(0, 100 - (offset_x / max_allowed_offset_x) * 50)
    y_score = max(0, 100 - (offset_y / max_allowed_offset_y) * 50)
    align_score = int(min(100, (x_score + y_score) / 2))

    if w < min_w:
        return "TOO FAR", False, "🔍 ATM SCAN: MOVE CLOSER TO CAMERA", align_score
    elif w > max_w:
        return "TOO CLOSE", False, "🔍 ATM SCAN: MOVE FARTHER BACK", align_score
    elif offset_x > max_allowed_offset_x or offset_y > max_allowed_offset_y:
        return "OFF CENTER", False, "🎯 ATM SCAN: CENTER YOUR FACE IN RING", align_score
    else:
        return "ALIGNED", True, "✓ ATM FACE ALIGNED - AUTHENTICATING...", 100


class AsyncFaceRecognizer(threading.Thread):
    """Background worker thread running DeepFace + ArcFace inference for ATM stream."""
    def __init__(self, ref_embedding, threshold=THRESHOLD, detector_backend=DETECTOR_BACKEND):
        super().__init__(daemon=True)
        self.ref_embedding = ref_embedding
        self.threshold = threshold
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
        self.status_color = (255, 191, 0)  # Amber
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

                try:
                    results = DeepFace.represent(
                        img_path=rgb_frame,
                        model_name="ArcFace",
                        detector_backend=self.detector_backend,
                        enforce_detection=False
                    )
                except Exception:
                    results = DeepFace.represent(
                        img_path=rgb_frame,
                        model_name="ArcFace",
                        detector_backend="opencv",
                        enforce_detection=False
                    )

                frame_h, frame_w = proc_frame.shape[:2]
                valid_faces = []
                for r in results:
                    area = r.get("facial_area", {})
                    w = area.get("w", 0)
                    h = area.get("h", 0)
                    if w > 0 and h > 0 and not (w >= frame_w and h >= frame_h):
                        valid_faces.append(r)

                with self.lock:
                    self.is_low_light = low_light_active
                    self.luminance_val = lum

                    if len(valid_faces) == 0:
                        self.face_status_str = "NOT DETECTED"
                        self.similarity_val = 0.0
                        self.similarity_str = "N/A"
                        self.match_status_str = "NO FACE"
                        self.current_bbox = None
                        self.status_color = (0, 0, 255)
                        self.multiple_faces_detected = False

                    elif len(valid_faces) > 1:
                        self.face_status_str = "MULTIPLE FACES"
                        self.similarity_val = 0.0
                        self.similarity_str = "N/A"
                        self.match_status_str = "MULTIPLE FACES"
                        self.current_bbox = None
                        self.status_color = (0, 0, 255)
                        self.multiple_faces_detected = True

                    else:
                        self.face_status_str = "DETECTED"
                        self.multiple_faces_detected = False
                        live_emb = valid_faces[0]["embedding"]
                        area = valid_faces[0]["facial_area"]
                        self.current_bbox = (area["x"], area["y"], area["w"], area["h"])

                        sim_val = float(cosine_similarity([self.ref_embedding], [live_emb])[0][0])
                        sim_pct = max(0.0, sim_val * 100)
                        self.similarity_val = sim_val
                        self.similarity_str = f"{sim_pct:.1f}% ({sim_val:.3f})"

                        if sim_val >= self.threshold:
                            self.match_status_str = "ACCOUNT MATCH ✓"
                            self.status_color = (0, 255, 127)  # Neon Green
                        else:
                            self.match_status_str = "NO MATCH"
                            self.status_color = (0, 0, 255)

            except Exception:
                with self.lock:
                    self.face_status_str = "NOT DETECTED"
                    self.similarity_val = 0.0
                    self.similarity_str = "N/A"
                    self.match_status_str = "NO FACE"
                    self.current_bbox = None
                    self.status_color = (0, 0, 255)
                    self.multiple_faces_detected = False

            self.frame_queue.task_done()


def draw_styled_box(img, x, y, w, h, color, scan_line_y=None):
    """Draw ATM biometric corner reticles and vertical laser scanline."""
    line_len = int(min(w, h) * 0.22)
    thickness = 2

    # Corners
    cv2.line(img, (x, y), (x + line_len, y), color, thickness)
    cv2.line(img, (x, y), (x, y + line_len), color, thickness)

    cv2.line(img, (x + w, y), (x + w - line_len, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + line_len), color, thickness)

    cv2.line(img, (x, y + h), (x + line_len, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - line_len), color, thickness)

    cv2.line(img, (x + w, y + h), (x + w - line_len, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - line_len), color, thickness)

    # Laser scan line animation
    if scan_line_y is not None:
        ly = y + int((scan_line_y % 100) / 100.0 * h)
        cv2.line(img, (x + 5, ly), (x + w - 5, ly), (255, 255, 255), 1)
        overlay = img.copy()
        cv2.rectangle(overlay, (x, max(y, ly - 4)), (x + w, min(y + h, ly + 4)), color, -1)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)


def draw_target_ring(img, center, axes, color, aligned=False):
    """Draw center ATM face target oval guide."""
    cv2.ellipse(img, center, axes, 0, 0, 360, (60, 60, 60), 1, cv2.LINE_AA)

    cx, cy = center
    ax, ay = axes
    cv2.line(img, (cx - ax - 10, cy), (cx - ax + 5, cy), color, 2)
    cv2.line(img, (cx + ax - 5, cy), (cx + ax + 10, cy), color, 2)
    cv2.line(img, (cx, cy - ay - 10), (cx, cy - ay + 5), color, 2)
    cv2.line(img, (cx, cy + ay - 5), (cx, cy + ay + 10), color, 2)

    if aligned:
        cv2.ellipse(img, center, axes, 0, 0, 360, color, 2, cv2.LINE_AA)


def draw_atm_success_modal(img, similarity_str, ref_name):
    """Draw ATM Biometric Authentication Success Screen."""
    h, w = img.shape[:2]

    # Dark frosted glass backdrop
    backdrop = img.copy()
    cv2.rectangle(backdrop, (0, 0), (w, h), (10, 20, 10), -1)
    cv2.addWeighted(backdrop, 0.80, img, 0.20, 0, img)

    # Central ATM modal panel
    pw, ph = int(w * 0.82), int(h * 0.52)
    px = (w - pw) // 2
    py = (h - ph) // 2

    # Draw ATM modal box
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (18, 28, 20), -1)
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (0, 255, 127), 2)
    cv2.rectangle(img, (px - 4, py - 4), (px + pw + 4, py + ph + 4), (0, 200, 100), 1)

    # Bank Icon Header
    chk_cx, chk_cy = px + pw // 2, py + 48
    cv2.circle(img, (chk_cx, chk_cy), 30, (0, 255, 127), -1)
    cv2.putText(img, "✓", (chk_cx - 14, chk_cy + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)

    # ATM Title & Account Details
    account_user = os.path.splitext(ref_name)[0]
    cv2.putText(img, "ATM BIOMETRIC VERIFICATION SUCCESSFUL!", (px + 20, py + 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    cv2.putText(img, f"ACCOUNT HOLDER : {account_user.upper()} (VERIFIED ✓)", (px + 40, py + 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 127), 2)

    cv2.putText(img, f"MATCH SIMILARITY : {similarity_str}  |  ATM TERMINAL #8092", (px + 40, py + 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 180), 1)

    cv2.putText(img, "You may now proceed to ATM Banking Services.", (px + 45, py + 225),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 255, 220), 1)

    # Action Button Prompt
    btn_y = py + ph - 45
    cv2.rectangle(img, (px + 40, btn_y), (px + pw - 40, btn_y + 35), (0, 255, 127), -1)
    cv2.putText(img, "PRESS ENTER / SPACE TO ACCESS ATM SERVICES", (px + 50, btn_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)


def main():
    print("=========================================================", flush=True)
    print("   BANK ATM FACIAL BIOMETRIC AUTHENTICATION TERMINAL", flush=True)
    print("=========================================================\n", flush=True)

    reference_image_path = find_reference_image()
    print(f"ATM Account Image Loaded:\n -> {reference_image_path}\n", flush=True)
    print(f"Initializing ATM Biometric Scanner ({DETECTOR_BACKEND})...\n", flush=True)

    ref_embedding, emb_dim = generate_reference_embedding(reference_image_path)

    print("\n=========================================================", flush=True)
    print(f"ATM Account Image : {reference_image_path}", flush=True)
    print(f"ArcFace Biometric : {emb_dim}-D vector active", flush=True)
    print(f"Match Threshold   : {THRESHOLD*100:.0f}% ({THRESHOLD})", flush=True)
    print("=========================================================\n", flush=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not access ATM camera module.", flush=True)
        print("⚠️ Make sure no other application (like web_app.py) is using the camera.\n", flush=True)
        sys.exit(1)

    print("✓ ATM Camera Module Active.\n", flush=True)
    print("Align your face in the ATM target ring.")
    print("Press Q to exit.\n", flush=True)

    recognizer_thread = AsyncFaceRecognizer(
        ref_embedding=ref_embedding,
        threshold=THRESHOLD,
        detector_backend=DETECTOR_BACKEND
    )
    recognizer_thread.start()

    window_title = "ATM Biometric Authentication Terminal #8092"

    scan_line_counter = 0
    verification_lock_count = 0
    is_verified_state = False
    chime_played = False
    camera_warmup_attempts = 0

    while True:
        ret, raw_frame = cap.read()
        if not ret or raw_frame is None:
            camera_warmup_attempts += 1
            if camera_warmup_attempts > 20:
                print("❌ Could not read frames from ATM camera.", flush=True)
                print("⚠️ Camera is busy or locked by another app (e.g., web_app.py). Please close web_app.py and try again.", flush=True)
                break
            time.sleep(0.05)
            continue
        
        camera_warmup_attempts = 0  # Reset once valid frame received

        raw_frame = cv2.flip(raw_frame, 1)
        frame_h, frame_w = raw_frame.shape[:2]

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
            display_frame, _, _ = enhance_low_light(raw_frame, threshold=LOW_LIGHT_THRESHOLD)
        else:
            display_frame = raw_frame.copy()

        align_status, is_aligned, align_prompt, align_score = check_face_alignment(
            current_bbox, frame_w, frame_h
        )

        if match_status_str == "ACCOUNT MATCH ✓" and is_aligned and not multiple_faces_detected:
            verification_lock_count += 1
            if verification_lock_count >= 4:
                is_verified_state = True
                if not chime_played:
                    play_atm_success_chime(customer_name=reference_image_path)
                    chime_played = True
        else:
            if not is_verified_state:
                verification_lock_count = max(0, verification_lock_count - 1)

        # ---------------------------------------------------------
        # Render ATM HUD UI
        # ---------------------------------------------------------
        center_pt = (frame_w // 2, frame_h // 2)
        target_axes = (int(frame_w * 0.16), int(frame_h * 0.26))
        ring_color = (0, 255, 127) if is_aligned else (0, 215, 255)
        draw_target_ring(display_frame, center_pt, target_axes, ring_color, aligned=is_aligned)

        scan_line_counter += 3
        if current_bbox is not None and not multiple_faces_detected:
            x, y, w, h = current_bbox
            box_color = (0, 255, 127) if is_verified_state else status_color
            draw_styled_box(display_frame, x, y, w, h, box_color, scan_line_y=scan_line_counter)

        if is_low_light:
            cv2.rectangle(display_frame, (0, 0), (frame_w, frame_h), (255, 255, 255), 4)

        # Top ATM Header
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_w, 70), (15, 20, 30), -1)
        cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)
        cv2.line(display_frame, (0, 70), (frame_w, 70), (40, 60, 80), 1)

        cv2.putText(display_frame, "BANK OF FUTURE | ATM BIOMETRIC SCANNER", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        if is_low_light:
            cv2.rectangle(display_frame, (frame_w - 240, 12), (frame_w - 20, 38), (0, 165, 255), -1)
            cv2.putText(display_frame, "🌙 ATM NIGHT SCAN BOOST", (frame_w - 235, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2)
        else:
            cv2.putText(display_frame, f"ATM LUX: {int(luminance_val)} (NORMAL)", (frame_w - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.putText(display_frame, f"MATCH: {similarity_str}", (20, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)

        cv2.putText(display_frame, f"ALIGNMENT: {align_score}% ({align_status})", (240, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, ring_color, 1)

        bar_x, bar_y, bar_w, bar_h = 440, 46, 120, 10
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * min(1.0, similarity_val))
        bar_fill_color = (0, 255, 127) if similarity_val >= THRESHOLD else (0, 0, 255)
        if fill_w > 0:
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_fill_color, -1)

        # Bottom ATM Action Bar
        bottom_overlay = display_frame.copy()
        cv2.rectangle(bottom_overlay, (0, frame_h - 50), (frame_w, frame_h), (15, 20, 30), -1)
        cv2.addWeighted(bottom_overlay, 0.75, display_frame, 0.25, 0, display_frame)

        if multiple_faces_detected:
            prompt = "⚠️ MULTIPLE FACES DETECTED - PLEASE KEEP ONLY 1 PERSON AT ATM"
            p_color = (0, 0, 255)
        elif is_verified_state:
            prompt = "✨ ATM BIOMETRIC VERIFIED - PROCEED TO BANKING SERVICES"
            p_color = (0, 255, 127)
        else:
            prompt = align_prompt
            p_color = ring_color

        cv2.putText(display_frame, prompt, (25, frame_h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, p_color, 2)

        if is_verified_state:
            draw_atm_success_modal(display_frame, similarity_str, reference_image_path)

        cv2.imshow(window_title, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        if is_verified_state and key in (13, 32):
            print("\n=========================================================", flush=True)
            print(" ✅ ATM USER AUTHENTICATED & PROCEEDED TO BANKING SERVICES!", flush=True)
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
