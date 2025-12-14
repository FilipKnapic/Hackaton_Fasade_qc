# micanje pozadine + yolo 11
# Instaliraj SVE potrebne pakete

# Download dataseta - YOLOv8 format je OK!
from roboflow import Roboflow
rf = Roboflow(api_key="HeccW424iHU818rXD9ly")
project = rf.workspace("myprojects-caukl").project("hackaton_only_negatives")
version = project.version(3)
dataset = version.download("yolov11")
# Treniranje modela
from ultralytics import YOLO
import os
import glob
from rembg import remove
from PIL import Image
import numpy as np

print("\n" + "="*60)
print("POKRETANJE TRENIRANJA MODELA - YOLO11 LARGE")
print("="*60 + "\n")

# YOLOv11 LARGE model - najjači prije xlarge!
model = YOLO('yolo11l.pt')

results = model.train(
    data=f'{dataset.location}/data.yaml',
    epochs=50,
    imgsz=640,
    batch=4,              # Još manji batch jer je model VELIKI
    name='fasade_yolo11_large',
    patience=20,
    save=True,
    plots=True,
    optimizer='AdamW',
    lr0=0.001,
    warmup_epochs=5,
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15,
    translate=0.1,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
)

print("\n" + "="*60)
print("TRENIRANJE ZAVRŠENO!")
print("="*60 + "\n")

# Učitaj najbolji trenirani model
best_model_path = 'runs/detect/fasade_yolo11_large/weights/best.pt'
best_model = YOLO(best_model_path)

# Validacija
print("Validacija modela...")
val_results = best_model.val()
print(f"mAP50: {val_results.box.map50:.3f}")
print(f"mAP50-95: {val_results.box.map:.3f}")
print(f"Precision: {val_results.box.mp:.3f}")
print(f"Recall: {val_results.box.mr:.3f}")

# FUNKCIJA ZA BACKGROUND REMOVAL
def remove_background(image_path, output_path):
    """Ukloni pozadinu sa slike"""
    input_img = Image.open(image_path)
    output_img = remove(input_img)
    output_img.save(output_path)
    return output_path

# Testiranje SA I BEZ background removela
print("\n" + "="*60)
print("IZVJEŠTAJ TESTIRANJA - YOLO11 LARGE + BACKGROUND REMOVAL")
print("="*60 + "\n")

test_images_path = f'{dataset.location}/test/images'
test_images = sorted(glob.glob(f'{test_images_path}/*'))

if len(test_images) == 0:
    test_images = sorted(glob.glob(f'{dataset.location}/valid/images/*'))
    print(f"Koristim validation slike: {len(test_images)}")

# Napravi folder za slike bez pozadine
nobg_folder = '/content/test_images_nobg_yolo11l'
os.makedirs(nobg_folder, exist_ok=True)

pass_count_original = 0
fail_count_original = 0
pass_count_nobg = 0
fail_count_nobg = 0

comparison_results = []

for img_path in test_images:
    img_name = os.path.basename(img_path)
    
    print(f"\n{'='*60}")
    print(f"Slika: {img_name}")
    print(f"{'='*60}")
    
    # 1. TEST NA ORIGINALNOJ SLICI
    pred_original = best_model.predict(img_path, verbose=False, conf=0.2)
    
    if len(pred_original[0].boxes) > 0:
        boxes = pred_original[0].boxes
        confidences = boxes.conf
        max_conf_idx = confidences.argmax()
        top_detection = boxes[max_conf_idx]
        
        class_id = int(top_detection.cls[0])
        class_name_original = pred_original[0].names[class_id]
        confidence_original = float(top_detection.conf[0])
    else:
        class_name_original = "no_detection"
        confidence_original = 0.0
    
    status_original = "✓ PASS" if class_name_original.lower() == "ok" else "✗ FAIL"
    if status_original == "✓ PASS":
        pass_count_original += 1
    else:
        fail_count_original += 1
    
    # 2. UKLONI POZADINU
    print("  Uklanjam pozadinu...")
    nobg_path = os.path.join(nobg_folder, img_name)
    remove_background(img_path, nobg_path)
    
    # 3. TEST NA SLICI BEZ POZADINE
    pred_nobg = best_model.predict(nobg_path, verbose=False, conf=0.2)
    
    if len(pred_nobg[0].boxes) > 0:
        boxes = pred_nobg[0].boxes
        confidences = boxes.conf
        max_conf_idx = confidences.argmax()
        top_detection = boxes[max_conf_idx]
        
        class_id = int(top_detection.cls[0])
        class_name_nobg = pred_nobg[0].names[class_id]
        confidence_nobg = float(top_detection.conf[0])
    else:
        class_name_nobg = "no_detection"
        confidence_nobg = 0.0
    
    status_nobg = "✓ PASS" if class_name_nobg.lower() == "ok" else "✗ FAIL"
    if status_nobg == "✓ PASS":
        pass_count_nobg += 1
    else:
        fail_count_nobg += 1
    
    # USPOREDBA
    improvement = "🔼 BOLJE" if confidence_nobg > confidence_original else "🔽 LOŠIJE" if confidence_nobg < confidence_original else "➡️ ISTO"
    
    print(f"\n  ORIGINALNA SLIKA:")
    print(f"    Status: {status_original}")
    print(f"    Klasa: {class_name_original}")
    print(f"    Confidence: {confidence_original:.2%}")
    
    print(f"\n  BEZ POZADINE:")
    print(f"    Status: {status_nobg}")
    print(f"    Klasa: {class_name_nobg}")
    print(f"    Confidence: {confidence_nobg:.2%}")
    
    print(f"\n  USPOREDBA: {improvement}")
    print(f"    Razlika u confidence: {(confidence_nobg - confidence_original)*100:+.1f}%")
    
    comparison_results.append({
        'image': img_name,
        'original_status': status_original,
        'nobg_status': status_nobg,
        'original_class': class_name_original,
        'nobg_class': class_name_nobg,
        'original_conf': confidence_original,
        'nobg_conf': confidence_nobg,
        'improvement': confidence_nobg - confidence_original
    })

# FINALNI IZVJEŠTAJ
print("\n" + "="*60)
print("FINALNI SAŽETAK - YOLO11 LARGE MODEL")
print("="*60)

print(f"\nUkupno testiranih slika: {len(test_images)}")

print(f"\n📊 ORIGINALNE SLIKE:")
print(f"  ✓ PASS: {pass_count_original} ({pass_count_original/len(test_images)*100:.1f}%)")
print(f"  ✗ FAIL: {fail_count_original} ({fail_count_original/len(test_images)*100:.1f}%)")

print(f"\n📊 SLIKE BEZ POZADINE:")
print(f"  ✓ PASS: {pass_count_nobg} ({pass_count_nobg/len(test_images)*100:.1f}%)")
print(f"  ✗ FAIL: {fail_count_nobg} ({fail_count_nobg/len(test_images)*100:.1f}%)")

# Koliko se poboljšalo
better_count = sum(1 for r in comparison_results if r['improvement'] > 0)
worse_count = sum(1 for r in comparison_results if r['improvement'] < 0)
same_count = sum(1 for r in comparison_results if r['improvement'] == 0)

print(f"\n📈 UTJECAJ BACKGROUND REMOVAL:")
print(f"  🔼 Poboljšano: {better_count} slika ({better_count/len(test_images)*100:.1f}%)")
print(f"  🔽 Pogoršano: {worse_count} slika ({worse_count/len(test_images)*100:.1f}%)")
print(f"  ➡️ Isto: {same_count} slika ({same_count/len(test_images)*100:.1f}%)")

avg_improvement = np.mean([r['improvement'] for r in comparison_results])
print(f"\n  Prosječno poboljšanje confidence: {avg_improvement*100:+.1f}%")

print("\n" + "="*60)
print(f"Slike bez pozadine: {nobg_folder}")
print(f"Model: {best_model_path}")
print("="*60)

# Prikaži rezultate vizualno
from IPython.display import Image as IPImage, display

print("\n📸 PRIKAZ PRVIH 5 REZULTATA:")
predict_dirs = glob.glob('runs/detect/predict*')
if predict_dirs:
    latest_predict = sorted(predict_dirs)[-1]
    result_images = sorted(glob.glob(f'{latest_predict}/*.jpg'))[:5]
    for img in result_images:
        print(f"\n{os.path.basename(img)}:")
        display(IPImage(filename=img, width=800))