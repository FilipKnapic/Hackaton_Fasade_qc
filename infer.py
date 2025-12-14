from ultralytics import YOLO
import os
import glob
from collections import Counter

# ==================================================
# 1. PUTANJE
# ==================================================
BASE_DIR = os.getcwd()

MODEL_PATH = os.path.join(BASE_DIR, "weights", "best.pt")
INPUT_DIR = os.path.join(BASE_DIR, "input_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")

os.makedirs(PRED_DIR, exist_ok=True)

assert os.path.exists(MODEL_PATH), "❌ best.pt nije pronađen"
assert os.path.exists(INPUT_DIR), "❌ input_images folder ne postoji"

# ==================================================
# 2. POSLOVNA LOGIKA
# ==================================================
ANOMALY_CLASSES = {
    "lim_nedostaje",
    "vijak_nedostaje",
    "vijak_na_pola_pritegnut",
    "vijak_labav",
    "staklo_puknuto",
    "brtva_ostecena",
}

CONF_THRESHOLD = 0.2

# ==================================================
# 3. MODEL
# ==================================================
model = YOLO(MODEL_PATH)
print("✅ Model učitan")

# ==================================================
# 4. UČITAJ SLIKE
# ==================================================
image_paths = []
for ext in ("*.jpg", "*.jpeg", "*.png"):
    image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

assert len(image_paths) > 0, "❌ Nema slika u input_images"
print(f"📸 Pronađeno slika: {len(image_paths)}")

# ==================================================
# 5. INFERENCE
# ==================================================
results_data = []
summary = Counter()

print("\n" + "=" * 70)
print("INFERENCE – QC MODE")
print("=" * 70)

for idx, img_path in enumerate(image_paths, 1):
    name = os.path.basename(img_path)

    results = model.predict(
        img_path,
        conf=CONF_THRESHOLD,
        save=True,
        project=OUTPUT_DIR,
        name="predictions",
        exist_ok=True,
        verbose=False
    )

    detected = []

    if len(results[0].boxes) > 0:
        for i in range(len(results[0].boxes)):
            cls_name = results[0].names[int(results[0].boxes.cls[i])]
            detected.append(cls_name)

    status = "PASS"
    defects = []

    for cls in detected:
        if cls in ANOMALY_CLASSES:
            status = "FAIL"
            defects.append(cls)

    summary[status] += 1

    results_data.append({
        "name": name,
        "status": status,
        "defects": defects,
        "image_path": f"predictions/{name}"
    })

    icon = "✅" if status == "PASS" else "❌"
    print(f"{icon} [{idx:03d}] {name:40} | {status}")

# ==================================================
# 6. HTML IZVJEŠĆE
# ==================================================
HTML_PATH = os.path.join(OUTPUT_DIR, "report.html")

html = """
<!DOCTYPE html>
<html lang="hr">
<head>
<meta charset="UTF-8">
<title>QC Izvješće</title>
<style>
body { font-family: Arial, sans-serif; background:#f5f6f8; padding:20px; }
h1 { margin-bottom:5px; }
table { width:100%; border-collapse:collapse; background:white; }
th, td { padding:10px; border-bottom:1px solid #ddd; text-align:left; vertical-align:top; }
th { background:#f0f0f0; }
.pass { color:#2e7d32; font-weight:bold; }
.fail { color:#c62828; font-weight:bold; }
img { max-width:400px; border:1px solid #ccc; border-radius:4px; }
.small { color:#666; font-size:0.9em; }
</style>
</head>
<body>

<h1>Quality Control Izvješće</h1>
<p class="small">YOLO11 – automatska evaluacija slika</p>

<table>
<tr>
<th>Slika</th>
<th>Status</th>
<th>Defekti</th>
<th>Detekcija</th>
</tr>
"""

for r in results_data:
    status_class = "pass" if r["status"] == "PASS" else "fail"
    defects = ", ".join(r["defects"]) if r["defects"] else "—"

    html += f"""
<tr>
<td>{r['name']}</td>
<td class="{status_class}">{r['status']}</td>
<td>{defects}</td>
<td><img src="{r['image_path']}" alt="{r['name']}"></td>
</tr>
"""

html += """
</table>
</body>
</html>
"""

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html)

# ==================================================
# 7. SAŽETAK
# ==================================================
print("\n" + "=" * 70)
print("SAŽETAK")
print("=" * 70)

for k, v in summary.items():
    print(f"{k:5} : {v}")

print(f"\n📄 HTML izvješće: {HTML_PATH}")
print("✅ GOTOVO – OTVORI report.html U BROWSERU")
