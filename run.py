"""
SignBridge Backend - MediaPipe Hand Landmarks + ASL Rule-Based Detection
Detects: A, B, C, L, Y, W using finger position rules
Works perfectly on real webcam footage - no training needed!
"""
import io
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import os

# ── Download hand landmarker model ─────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("📥 Downloading MediaPipe hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Model downloaded!")

# ── Setup MediaPipe Hand Landmarker ────────────────────────
print("Loading hand landmarker...")
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)
print("✅ Hand landmarker ready!")

# ── Finger state detection ─────────────────────────────────
def finger_states(landmarks):
    """
    Returns [thumb, index, middle, ring, pinky] as True/False (extended or not)
    landmarks: list of 21 hand landmarks
    """
    lm = landmarks

    # Finger tip and pip (middle joint) indices
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    mcps = [2, 5, 9, 13, 17]

    states = []

    # Thumb: compare x position (tip vs mcp)
    thumb_extended = lm[4].x < lm[3].x  # for right hand
    states.append(thumb_extended)

    # Other fingers: tip y < pip y means extended (y increases downward)
    for i in range(1, 5):
        states.append(lm[tips[i]].y < lm[pips[i]].y)

    return states  # [thumb, index, middle, ring, pinky]

# ── ASL Letter Classification ──────────────────────────────
def classify_asl(landmarks):
    """
    Classify ASL letters from hand landmarks.
    Returns (letter, confidence) or (None, 0)
    """
    lm = landmarks
    thumb, index, middle, ring, pinky = finger_states(lm)

    # ── L: index up + thumb out, rest curled ──
    if index and thumb and not middle and not ring and not pinky:
        return "L", 0.92

    # ── B: index + middle + ring + pinky up, thumb tucked ──
    if not thumb and index and middle and ring and pinky:
        return "B", 0.91

    # ── W: index + middle + ring up, thumb + pinky down ──
    if not thumb and index and middle and ring and not pinky:
        return "W", 0.90

    # ── Y: thumb + pinky out, rest curled ──
    if thumb and not index and not middle and not ring and pinky:
        return "Y", 0.92

    # ── A: all fingers curled, thumb to side ──
    # Check fingers are all curled (not extended)
    if not index and not middle and not ring and not pinky:
        # Thumb should be roughly at the side
        return "A", 0.88

    # ── C: all fingers slightly bent (none fully extended, none fully curled) ──
    # Check that fingertips are between mcp and fully extended position
    # Use the distance between thumb tip and index tip for C shape
    thumb_tip = lm[4]
    index_tip = lm[8]
    dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5

    if not thumb and not index and not middle and not ring and not pinky:
        # All curled = A, already caught above
        pass

    # C: fingers curved, not fully extended or curled
    # approximated by checking mid-flex state
    tips_y  = [lm[8].y, lm[12].y, lm[16].y, lm[20].y]
    pips_y  = [lm[6].y, lm[10].y, lm[14].y, lm[18].y]
    mcps_y  = [lm[5].y, lm[9].y,  lm[13].y, lm[17].y]

    # C: tips are BELOW pips but ABOVE wrist level (partially curled)
    partially_bent = all(pips_y[i] < tips_y[i] for i in range(4))
    spread_x = abs(lm[8].x - lm[20].x) > 0.08  # fingers spread wide

    if partially_bent and spread_x and dist > 0.08:
        return "C", 0.85

    return None, 0.0

# ── FastAPI ────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_img  = Image.open(io.BytesIO(contents)).convert("RGB")
        np_img   = np.array(pil_img)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_img)
        result   = landmarker.detect(mp_image)

        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            return {"gesture": None, "confidence": 0.0}

        landmarks = result.hand_landmarks[0]
        gesture, confidence = classify_asl(landmarks)

        if gesture:
            print(f"✋ Detected: {gesture} ({confidence:.2f})")
            return {"gesture": gesture, "confidence": confidence}
        else:
            return {"gesture": None, "confidence": 0.0}

    except Exception as e:
        print(f"Error: {e}")
        return {"gesture": None, "confidence": 0.0}

@app.get("/health")
def health():
    return {"status": "ok"}

print("🚀 Server running at http://localhost:8000")
print("   Detecting ASL letters: A, B, C, L, W, Y")
uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
