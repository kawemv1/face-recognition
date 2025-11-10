
import cv2, numpy as np, os
from tqdm import tqdm
from insightface.app import FaceAnalysis

# ===============================
# 1️⃣ Setup
# ===============================
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))  # use CPU (-1) or GPU (0)

# ===============================
# 2️⃣ Load known faces
# ===============================
known_embs, known_names = [], []
os.makedirs("faces/known", exist_ok=True)
for f in os.listdir("faces/known"):
    if not f.lower().endswith((".jpg", ".png")):
        continue
    img = cv2.imread(os.path.join("faces/known", f))
    faces = app.get(img)
    if faces:
        known_embs.append(faces[0].normed_embedding)
        known_names.append(os.path.splitext(f)[0])
        print(f"✅ Loaded {f}")
print(f"📚 Total known faces: {len(known_names)}")

# ===============================
# 3️⃣ Choose video
# ===============================
video_file = "input.mp4"  # replace this with your video file
cap = cv2.VideoCapture(video_file)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    "output.mp4",
    fourcc,
    cap.get(cv2.CAP_PROP_FPS),
    (int(cap.get(3)), int(cap.get(4))),
)

# ===============================
# 4️⃣ Process video frames
# ===============================
frame_skip = 3
count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("🚀 Processing video...")

for _ in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % frame_skip != 0:
        continue

    faces = app.get(frame)
    for face in faces:
        emb = face.normed_embedding
        if known_embs:
            sims = [np.dot(emb, k) for k in known_embs]
            idx = int(np.argmax(sims))
            name, conf = known_names[idx], sims[idx]
        else:
            name, conf = "Unknown", 0.0

        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
        cv2.putText(
            frame,
            f"{name} ({conf:.2f})",
            (box[0], box[1]-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )
    out.write(frame)

cap.release()
out.release()
print("✅ Done! Saved as output.mp4")
