import os
import sys
import base64
import webbrowser
import threading
import time

# Ensure UTF-8 output encoding & unbuffered stdout for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Silence TensorFlow / DeepFace verbose logging before importing deepface
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

import subprocess

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

app = Flask(__name__)

DAY_THRESHOLD = 0.58    # Daylight cosine matching threshold
NIGHT_THRESHOLD = 0.48  # Night vision adaptive cosine matching threshold
DETECTOR_BACKEND = "mtcnn"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
LOW_LIGHT_THRESHOLD = 90.0  # Ambient luminance threshold for auto night-vision boost

REFERENCE_EMBEDDINGS = {}  # { filename: [emb_standard, emb_night, emb_gray] }
ACTIVE_REFERENCE_NAME = None
LAST_AUDIO_TIME = 0


def play_atm_success_chime(customer_name="Customer"):
    """Play ATM chime sound and speak voice confirmation aloud in a background thread."""
    global LAST_AUDIO_TIME
    now = time.time()
    if now - LAST_AUDIO_TIME < 6.0:
        return
    LAST_AUDIO_TIME = now

    def audio_worker():
        try:
            if HAS_WINSOUND:
                winsound.Beep(523, 90)
                time.sleep(0.01)
                winsound.Beep(659, 90)
                time.sleep(0.01)
                winsound.Beep(784, 110)
                time.sleep(0.01)
                winsound.Beep(1046, 240)
                time.sleep(0.15)

            raw_base = os.path.splitext(os.path.basename(str(customer_name)))[0]
            clean_name = raw_base.split('.')[-1].strip().capitalize()

            print("\n=========================================================", flush=True)
            print(f" ✅ ATM USER {customer_name.upper()} AUTHENTICATED & VERIFIED!", flush=True)
            print("=========================================================\n", flush=True)

            speech_text = f"Customer {clean_name} has been successfully verified. You may now proceed to ATM banking services."
            cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{speech_text}\');"'
            subprocess.run(cmd, shell=True, capture_output=True)
        except Exception:
            pass

    threading.Thread(target=audio_worker, daemon=True).start()


def select_reference_image():
    """
    Search directory for ATM reference account face images (.jpg, .jpeg, .png).
    Prompt user in terminal if multiple images exist.
    """
    current_dir = os.getcwd()
    found_images = [
        f for f in os.listdir(current_dir)
        if os.path.isfile(os.path.join(current_dir, f)) and f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    if len(found_images) == 0:
        print("❌ No ATM account reference image found.\n", flush=True)
        print("Please place your account face image in the project folder.", flush=True)
        sys.exit(1)
    elif len(found_images) == 1:
        selected_image = found_images[0]
        print(f"ATM Account Image Found:\n 1. {selected_image}\n", flush=True)
        return selected_image, found_images
    else:
        print("Multiple ATM Account Images Found:\n", flush=True)
        for idx, img_name in enumerate(found_images, 1):
            print(f" {idx}. {img_name}", flush=True)
        print(flush=True)
        while True:
            try:
                sys.stdout.flush()
                choice = input("Enter the number of your ATM Account Image: ").strip()
                choice_num = int(choice)
                if 1 <= choice_num <= len(found_images):
                    selected_image = found_images[choice_num - 1]
                    print(f"\nATM Account Image Selected:\n -> {selected_image}\n", flush=True)
                    return selected_image, found_images
                else:
                    print(f"Please enter a number between 1 and {len(found_images)}.", flush=True)
            except (ValueError, KeyboardInterrupt, EOFError):
                print("Invalid selection. Exiting.", flush=True)
                sys.exit(1)


def enhance_low_light(frame, threshold=LOW_LIGHT_THRESHOLD):
    """
    Multi-Stage Low-Light Auto-Brightness Enhancement Engine.
    Combines Dynamic Range Percentile Contrast Stretching, Adaptive CLAHE Equalization,
    Power-Law Gamma LUT, Bilateral Denoising, and Unsharp Sharpening.
    """
    if frame is None:
        return frame, False, 0.0

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    mean_lum = float(np.mean(l_channel))

    is_low_light = mean_lum < threshold

    if is_low_light:
        p_low, p_high = np.percentile(l_channel, (1, 99))
        if p_high > p_low:
            l_stretched = np.clip((l_channel.astype(np.float32) - p_low) * (255.0 / (p_high - p_low + 1e-5)), 0, 255).astype(np.uint8)
        else:
            l_stretched = l_channel.copy()

        clip_limit = 6.0 if mean_lum < 45.0 else 4.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l_stretched)

        gamma = max(0.20, min(3.8, 160.0 / (mean_lum + 1e-5)))
        inv_gamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        cl = cv2.LUT(cl, lut)

        denoised = cv2.bilateralFilter(cl, d=5, sigmaColor=35, sigmaSpace=35)
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
        sharpened = cv2.addWeighted(denoised, 1.4, gaussian, -0.4, 0)

        enhanced_lab = cv2.merge((sharpened, a, b))
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_frame, True, mean_lum

    return frame, False, mean_lum


def generate_multi_embeddings_for_image(img_name):
    """
    Generate Multi-Domain ArcFace 512-D vectors (Standard, Night-Vision, Equalized)
    for high-accuracy matching in all illumination conditions.
    """
    embeddings = []

    # 1. Standard RGB
    try:
        results = DeepFace.represent(
            img_path=img_name,
            model_name="ArcFace",
            detector_backend=DETECTOR_BACKEND,
            align=True,
            enforce_detection=True
        )
        valid = [r for r in results if r.get("facial_area", {}).get("w", 0) > 40]
        if valid:
            embeddings.append(valid[0]["embedding"])
    except Exception:
        try:
            results = DeepFace.represent(
                img_path=img_name,
                model_name="ArcFace",
                detector_backend="opencv",
                align=True,
                enforce_detection=True
            )
            valid = [r for r in results if r.get("facial_area", {}).get("w", 0) > 40]
            if valid:
                embeddings.append(valid[0]["embedding"])
        except Exception:
            pass

    # 2. Night-Enhanced Domain
    try:
        raw_img = cv2.imread(img_name)
        if raw_img is not None:
            enhanced_img, _, _ = enhance_low_light(raw_img, threshold=255.0)
            enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
            results = DeepFace.represent(
                img_path=enhanced_rgb,
                model_name="ArcFace",
                detector_backend="opencv",
                align=True,
                enforce_detection=False
            )
            valid = [r for r in results if r.get("facial_area", {}).get("w", 0) > 40]
            if valid:
                embeddings.append(valid[0]["embedding"])
    except Exception:
        pass

    # 3. Grayscale Equalized Domain
    try:
        raw_img = cv2.imread(img_name)
        if raw_img is not None:
            gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            eq = cv2.equalizeHist(gray)
            eq_bgr = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
            eq_rgb = cv2.cvtColor(eq_bgr, cv2.COLOR_BGR2RGB)
            results = DeepFace.represent(
                img_path=eq_rgb,
                model_name="ArcFace",
                detector_backend="opencv",
                align=True,
                enforce_detection=False
            )
            valid = [r for r in results if r.get("facial_area", {}).get("w", 0) > 40]
            if valid:
                embeddings.append(valid[0]["embedding"])
    except Exception:
        pass

    return embeddings


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


def init_reference_system():
    global REFERENCE_EMBEDDINGS, ACTIVE_REFERENCE_NAME
    selected_img, all_found_images = select_reference_image()
    ACTIVE_REFERENCE_NAME = selected_img

    print(f"Generating Multi-Domain ArcFace embeddings for ATM account image '{selected_img}'...", flush=True)
    embs = generate_multi_embeddings_for_image(selected_img)
    if not embs:
        print(f"❌ Could not detect face in ATM reference image '{selected_img}'.", flush=True)
        sys.exit(1)

    REFERENCE_EMBEDDINGS[selected_img] = embs
    print(f"✓ ATM Reference Face '{selected_img}' loaded with {len(embs)} Multi-Domain ArcFace vectors!\n", flush=True)

    for img_name in all_found_images:
        if img_name not in REFERENCE_EMBEDDINGS:
            other_embs = generate_multi_embeddings_for_image(img_name)
            if other_embs:
                REFERENCE_EMBEDDINGS[img_name] = other_embs

    return True


@app.route("/")
def index():
    available_images = list(REFERENCE_EMBEDDINGS.keys())
    account_user = os.path.splitext(ACTIVE_REFERENCE_NAME)[0] if ACTIVE_REFERENCE_NAME else "Unknown"
    active_idx = available_images.index(ACTIVE_REFERENCE_NAME) + 1 if (ACTIVE_REFERENCE_NAME and ACTIVE_REFERENCE_NAME in available_images) else 1
    return render_template(
        "index.html",
        active_reference=ACTIVE_REFERENCE_NAME or "None",
        account_user=f"{active_idx}. {account_user.upper()}",
        available_images=available_images
    )


@app.route("/set_reference", methods=["POST"])
def set_reference():
    global ACTIVE_REFERENCE_NAME
    data = request.get_json()
    img_name = data.get("image_name")
    if img_name in REFERENCE_EMBEDDINGS:
        ACTIVE_REFERENCE_NAME = img_name
        available_images = list(REFERENCE_EMBEDDINGS.keys())
        idx = available_images.index(img_name) + 1
        account_user = os.path.splitext(ACTIVE_REFERENCE_NAME)[0]
        return jsonify({
            "success": True,
            "active_reference": ACTIVE_REFERENCE_NAME,
            "account_user": f"{idx}. {account_user.upper()}"
        })
    return jsonify({"success": False, "error": "Image not found."})


@app.route("/verify_frame", methods=["POST"])
def verify_frame():
    global ACTIVE_REFERENCE_NAME, REFERENCE_EMBEDDINGS
    if not ACTIVE_REFERENCE_NAME or ACTIVE_REFERENCE_NAME not in REFERENCE_EMBEDDINGS:
        return jsonify({"success": False, "error": "No reference face loaded."})

    ref_embs = REFERENCE_EMBEDDINGS[ACTIVE_REFERENCE_NAME]

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"success": False, "error": "No frame received."})

    try:
        img_data = data["image"].split(",")[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "error": "Invalid frame decode."})

        frame_h, frame_w = frame.shape[:2]

        proc_frame, is_low_light, luminance_val = enhance_low_light(frame)
        rgb_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)

        # Adaptive threshold based on lighting conditions
        current_threshold = NIGHT_THRESHOLD if is_low_light else DAY_THRESHOLD

        # Multi-Detector Cascade Fallback for Dark / Low-Light Conditions
        backends_to_try = [DETECTOR_BACKEND, "opencv", "ssd"] if is_low_light else [DETECTOR_BACKEND, "opencv"]
        valid_faces = []

        for backend in backends_to_try:
            try:
                results = DeepFace.represent(
                    img_path=rgb_frame,
                    model_name="ArcFace",
                    detector_backend=backend,
                    align=True,
                    enforce_detection=False
                )
                for r in results:
                    area = r.get("facial_area", {})
                    w = area.get("w", 0)
                    h = area.get("h", 0)
                    if w > 25 and h > 25 and not (w >= frame_w and h >= frame_h):
                        valid_faces.append(r)
                if valid_faces:
                    break
            except Exception:
                continue

        if not valid_faces:
            return jsonify({
                "success": True,
                "face_detected": False,
                "is_low_light": is_low_light,
                "luminance": int(luminance_val),
                "message": "NO FACE DETECTED"
            })

        if len(valid_faces) > 1:
            return jsonify({
                "success": True,
                "face_detected": True,
                "multiple_faces": True,
                "is_low_light": is_low_light,
                "luminance": int(luminance_val),
                "message": "MULTIPLE FACES DETECTED"
            })

        target_face = valid_faces[0]
        live_emb = target_face["embedding"]
        bbox = target_face["facial_area"]

        sim_val = compute_best_cosine_similarity(ref_embs, live_emb)
        sim_pct = max(0.0, sim_val * 100)
        is_match = sim_val >= current_threshold

        available_images = list(REFERENCE_EMBEDDINGS.keys())
        active_idx = available_images.index(ACTIVE_REFERENCE_NAME) + 1 if ACTIVE_REFERENCE_NAME in available_images else 1
        account_user_clean = os.path.splitext(ACTIVE_REFERENCE_NAME)[0]

        if is_match:
            play_atm_success_chime(account_user_clean)

        other_matched_user = None
        if not is_match and REFERENCE_EMBEDDINGS:
            for img_name, other_embs in REFERENCE_EMBEDDINGS.items():
                if img_name != ACTIVE_REFERENCE_NAME:
                    other_sim = compute_best_cosine_similarity(other_embs, live_emb)
                    if other_sim >= current_threshold:
                        o_idx = available_images.index(img_name) + 1 if img_name in available_images else 1
                        o_name = os.path.splitext(img_name)[0].upper()
                        other_matched_user = f"{o_idx}. {o_name}"
                        break

        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        fcx = x + w / 2.0
        fcy = y + h / 2.0
        cx, cy = frame_w / 2.0, frame_h / 2.0
        offset_x, offset_y = abs(fcx - cx), abs(fcy - cy)

        max_off_x, max_off_y = frame_w * 0.20, frame_h * 0.22
        is_aligned = (
            (frame_w * 0.16 <= w <= frame_w * 0.60) and
            (offset_x <= max_off_x) and
            (offset_y <= max_off_y)
        )

        return jsonify({
            "success": True,
            "face_detected": True,
            "multiple_faces": False,
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "similarity": round(sim_pct, 1),
            "sim_val": round(sim_val, 3),
            "is_match": is_match,
            "is_aligned": is_aligned,
            "is_low_light": is_low_light,
            "luminance": int(luminance_val),
            "active_reference": ACTIVE_REFERENCE_NAME,
            "account_user": f"{active_idx}. {account_user_clean.upper()}",
            "clean_customer_name": account_user_clean.capitalize(),
            "other_matched_user": other_matched_user,
            "message": "MATCH VERIFIED" if is_match else "USER NOT AUTHENTICATED OR MATCHED"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


def open_browser():
    time.sleep(1.5)
    print("\n=========================================================", flush=True)
    print(" 🚀 Opening ATM Biometric Web Terminal at http://localhost:5000 ...", flush=True)
    print("=========================================================\n", flush=True)
    webbrowser.open("http://localhost:5000")


if __name__ == "__main__":
    print("=========================================================", flush=True)
    print("    BANK OF FUTURE - ATM FACIAL BIOMETRIC TERMINAL", flush=True)
    print("=========================================================\n", flush=True)
    if init_reference_system():
        threading.Thread(target=open_browser, daemon=True).start()
        print("Starting ATM Flask Web Server on http://localhost:5000...", flush=True)
        app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        print("❌ Could not initialize ATM reference face system.", flush=True)
