"""
╔══════════════════════════════════════════════════════════╗
║           SignBridge — Complete Backend                  ║
║                                                          ║
║  SETUP (run these once):                                 ║
║                                                          ║
║  pip install fastapi uvicorn opencv-python               ║
║              torch torchvision numpy pandas              ║
║                                                          ║
║  Download dataset from Kaggle:                           ║
║  kaggle.com/datasets/datamunge/sign-language-mnist       ║
║  → put sign_mnist_train.csv + sign_mnist_test.csv        ║
║    inside a folder called  data/                         ║
║                                                          ║
║  THEN RUN:                                               ║
║  python backend.py --train    (first time only ~2 min)   ║
║  python backend.py            (every time after)         ║
║                                                          ║
║  Open index.html in Chrome. Done ✅                      ║
╚══════════════════════════════════════════════════════════╝
"""

import sys, os, threading
import cv2, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ══════════════════════════════════════════════════════════
#  MODEL  (your friend's CNN — unchanged)
# ══════════════════════════════════════════════════════════

class SignCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 5 * 5, 128)
        self.fc2   = nn.Linear(128, 26)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ══════════════════════════════════════════════════════════
#  LABEL MAP
#  Sign MNIST = A-Y, no J or Z (they require motion)
# ══════════════════════════════════════════════════════════

LABEL_MAP = {
    0:"A",  1:"B",  2:"C",  3:"D",  4:"E",
    5:"F",  6:"G",  7:"H",  8:"I",
    9:"K",  10:"L", 11:"M", 12:"N", 13:"O",
    14:"P", 15:"Q", 16:"R", 17:"S", 18:"T",
    19:"U", 20:"V", 21:"W", 22:"X", 23:"Y",
}

MODEL_PATH = "sign_model.pth"
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════
#  TRAINING  —  python backend.py --train
# ══════════════════════════════════════════════════════════

def train():
    print("📚  Loading dataset...")
    if not os.path.exists("data/sign_mnist_train.csv"):
        print("\n❌  data/sign_mnist_train.csv not found!")
        print("    Download: kaggle.com/datasets/datamunge/sign-language-mnist")
        sys.exit(1)

    def load_csv(path):
        df = pd.read_csv(path)
        X  = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32) / 255.0
        y  = torch.tensor(df.iloc[:, 0].values,  dtype=torch.long)
        return X.view(-1, 1, 28, 28), y

    X_tr, y_tr = load_csv("data/sign_mnist_train.csv")
    X_te, y_te = load_csv("data/sign_mnist_test.csv")
    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_te, y_te = X_te.to(device), y_te.to(device)

    model = SignCNN().to(device)
    opt   = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    print(f"🧠  Training on {device} — 10 epochs\n")
    for ep in range(10):
        model.train()
        loss = loss_fn(model(X_tr), y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        print(f"    Epoch {ep+1:02d}/10  loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        acc = (torch.argmax(model(X_te), 1) == y_te).float().mean().item()
    print(f"\n✅  Accuracy: {acc*100:.1f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾  Saved → {MODEL_PATH}")
    print("\nNow run:  python backend.py\n")


# ══════════════════════════════════════════════════════════
#  PREPROCESSING
#  Webcam frame → 28×28 grayscale (matches Sign MNIST)
# ══════════════════════════════════════════════════════════

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    s = min(h, w)
    crop = gray[(h-s)//2:(h-s)//2+s, (w-s)//2:(w-s)//2+s]
    img  = cv2.resize(crop, (28, 28))
    t    = torch.tensor(img, dtype=torch.float32) / 255.0
    return t.unsqueeze(0).unsqueeze(0).to(device)   # (1,1,28,28)


# ══════════════════════════════════════════════════════════
#  CAMERA THREAD
#  Reads webcam, runs CNN, writes result to shared state
# ══════════════════════════════════════════════════════════

current    = {"gesture": None, "confidence": 0.0}
state_lock = threading.Lock()

def camera_loop(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️   Cannot open webcam — is another app using it?")
        return

    print("📷  Webcam open. AI is recognising signs...\n")
    THRESHOLD = 0.55
    n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        n += 1
        if n % 2 != 0:   # run every other frame to save CPU
            continue

        try:
            t = preprocess(frame)
            with torch.no_grad():
                probs      = F.softmax(model(t), dim=1)
                conf, pred = torch.max(probs, 1)
                conf_val   = conf.item()
                gesture    = LABEL_MAP.get(pred.item())

            with state_lock:
                if gesture and conf_val >= THRESHOLD:
                    current["gesture"]    = gesture
                    current["confidence"] = round(conf_val, 3)
                else:
                    current["gesture"]    = None
                    current["confidence"] = 0.0

        except Exception as e:
            print(f"Inference error: {e}")

    cap.release()


# ══════════════════════════════════════════════════════════
#  FASTAPI  —  frontend polls GET /gesture every 500ms
# ══════════════════════════════════════════════════════════

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/gesture")
def get_gesture():
    with state_lock:
        return dict(current)

@app.get("/health")
def health():
    return {"status": "ok"}


# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":

    if "--train" in sys.argv:
        train()
        sys.exit(0)

    if not os.path.exists(MODEL_PATH):
        print(f"\n❌  {MODEL_PATH} not found — train first:")
        print("    python backend.py --train\n")
        sys.exit(1)

    model = SignCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    print(f"✅  Model loaded ({device})")

    threading.Thread(target=camera_loop, args=(model,), daemon=True).start()

    print("🚀  API → http://localhost:8000")
    print("    Open index.html in Chrome and start signing!\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
