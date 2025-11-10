import cv2, os
src = "faces/known"
for f in os.listdir(src):
    if f.lower().endswith((".png",".jpeg",".jpg")):
        p = os.path.join(src,f)
        img = cv2.imread(p, cv2.IMREAD_COLOR)     # always 8-bit BGR
        cv2.imwrite(p, img)                       # overwrite as 8-bit jpg/png
        print("✅ Converted", f)
