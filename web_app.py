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

app = Flask(__name__)

THRESHOLD = 0.60
DETECTOR_BACKEND = "mtcnn"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
LOW_LIGHT_THRESHOLD = 75.0

REFERENCE_EMBEDDINGS = {}  # { filename: embedding }
ACTIVE_REFERENCE_NAME = None


def select_reference_image():
    """
    Search directory for ATM reference account face images (.jpg, .jpeg, .png).
    Prompt user in terminal to enter the number.
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
        print(f"ATM Account Image Found:\n{selected_image}\n", flush=True)
        return selected_image, found_images
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
                    selected_image = found_images[choice_num - 1]
                    print(f"\nATM Account Image Selected:\n -> {selected_image}\n", flush=True)
                    return selected_image, found_images
                else:
                    print(f"Please enter a number between 1 and {len(found_images)}.", flush=True)
            except (ValueError, KeyboardInterrupt, EOFError):
                print("Invalid selection. Exiting.", flush=True)
                sys.exit(1)


def generate_embedding_for_image(img_name):
    """Generate ArcFace embedding for a specific ATM account image."""
    try:
        results = DeepFace.represent(
            img_path=img_name,
            model_name="ArcFace",
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )
        valid = [r for r in results if r.get("facial_area", {}).get("w", 0) > 60]
        if valid:
            return valid[0]["embedding"]
    except Exception:
        try:
            results = DeepFace.represent(
                img_path=img_name,
                model_name="ArcFace",
                detector_backend="opencv",
                enforce_detection=True
            )
            valid = [r for r in results if r.get("facial_area", {}).get("w", 0) > 60]
            if valid:
                return valid[0]["embedding"]
        except Exception as ex:
            print(f" ⚠️ Could not embed '{img_name}': {ex}", flush=True)
    return None


def init_reference_system():
    global REFERENCE_EMBEDDINGS, ACTIVE_REFERENCE_NAME
    selected_img, all_found_images = select_reference_image()
    ACTIVE_REFERENCE_NAME = selected_img

    print(f"Generating ArcFace embedding for ATM account image '{selected_img}'...", flush=True)
    emb = generate_embedding_for_image(selected_img)
    if emb is None:
        print(f"❌ Could not detect face in ATM reference image '{selected_img}'.", flush=True)
        sys.exit(1)

    REFERENCE_EMBEDDINGS[selected_img] = emb
    print(f"✓ ATM Reference Face '{selected_img}' embedding generated successfully!\n", flush=True)

    for img_name in all_found_images:
        if img_name not in REFERENCE_EMBEDDINGS:
            other_emb = generate_embedding_for_image(img_name)
            if other_emb is not None:
                REFERENCE_EMBEDDINGS[img_name] = other_emb

    return True


def enhance_low_light(frame, threshold=LOW_LIGHT_THRESHOLD):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    mean_lum = float(np.mean(l_channel))

    is_low_light = mean_lum < threshold

    if is_low_light:
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)

        gamma = max(0.4, min(2.5, 130.0 / (mean_lum + 1e-5)))
        inv_gamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        cl = cv2.LUT(cl, lut)

        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_frame, True, mean_lum

    return frame, False, mean_lum


@app.route("/")
def index():
    available_images = list(REFERENCE_EMBEDDINGS.keys())
    account_user = os.path.splitext(ACTIVE_REFERENCE_NAME)[0] if ACTIVE_REFERENCE_NAME else "Unknown"
    return render_template(
        "index.html",
        active_reference=ACTIVE_REFERENCE_NAME or "None",
        account_user=account_user.upper(),
        available_images=available_images
    )


@app.route("/set_reference", methods=["POST"])
def set_reference():
    global ACTIVE_REFERENCE_NAME
    data = request.get_json()
    img_name = data.get("image_name")
    if img_name in REFERENCE_EMBEDDINGS:
        ACTIVE_REFERENCE_NAME = img_name
        account_user = os.path.splitext(ACTIVE_REFERENCE_NAME)[0]
        return jsonify({
            "success": True,
            "active_reference": ACTIVE_REFERENCE_NAME,
            "account_user": account_user.upper()
        })
    return jsonify({"success": False, "error": "Image not found."})


@app.route("/verify_frame", methods=["POST"])
def verify_frame():
    global ACTIVE_REFERENCE_NAME, REFERENCE_EMBEDDINGS
    if not ACTIVE_REFERENCE_NAME or ACTIVE_REFERENCE_NAME not in REFERENCE_EMBEDDINGS:
        return jsonify({"success": False, "error": "No reference face loaded."})

    ref_emb = REFERENCE_EMBEDDINGS[ACTIVE_REFERENCE_NAME]

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

        try:
            results = DeepFace.represent(
                img_path=rgb_frame,
                model_name="ArcFace",
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False
            )
        except Exception:
            results = DeepFace.represent(
                img_path=rgb_frame,
                model_name="ArcFace",
                detector_backend="opencv",
                enforce_detection=False
            )

        valid_faces = []
        for r in results:
            area = r.get("facial_area", {})
            w = area.get("w", 0)
            h = area.get("h", 0)
            if w > 0 and h > 0 and not (w >= frame_w and h >= frame_h):
                valid_faces.append(r)

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

        sim_val = float(cosine_similarity([ref_emb], [live_emb])[0][0])
        sim_pct = max(0.0, sim_val * 100)
        is_match = sim_val >= THRESHOLD

        other_matched_user = None
        if not is_match and REFERENCE_EMBEDDINGS:
            for img_name, emb in REFERENCE_EMBEDDINGS.items():
                if img_name != ACTIVE_REFERENCE_NAME:
                    other_sim = float(cosine_similarity([emb], [live_emb])[0][0])
                    if other_sim >= THRESHOLD:
                        other_matched_user = os.path.splitext(img_name)[0].upper()
                        break

        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        fcx = x + w / 2.0
        fcy = y + h / 2.0
        cx, cy = frame_w / 2.0, frame_h / 2.0
        offset_x, offset_y = abs(fcx - cx), abs(fcy - cy)

        max_off_x, max_off_y = frame_w * 0.18, frame_h * 0.20
        is_aligned = (
            (frame_w * 0.18 <= w <= frame_w * 0.58) and
            (offset_x <= max_off_x) and
            (offset_y <= max_off_y)
        )

        account_user = os.path.splitext(ACTIVE_REFERENCE_NAME)[0]

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
            "account_user": account_user.upper(),
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
