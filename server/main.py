from constants import (
    CLASS_NAMES,
    COIN_VALUES,
    COLOURS,
    DETECTION_CONF_THRESHOLD,
    DETECTION_THRESHOLD,
    IDENTICAL_IOU_THRESHOLD,
    INPUT_IMG_RATIO,
    INPUT_IMG_SIZE,
    MODEL_PATH,
    RESULTS_FOLDER,
    UPLOAD_FOLDER,
)
from utils import allowed_file, bb_intersection_over_union
import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont, ExifTags
import glob

model = YOLO(MODEL_PATH)


for ORIENTATION_KEY in ExifTags.TAGS.keys():
    if ExifTags.TAGS[ORIENTATION_KEY] == "Orientation":
        break

def process_image(path):
    img = Image.open(path)

    exif = img.getexif()
            
    # Rotate
    if len(exif) > 0:
        if exif[ORIENTATION_KEY] == 3:
            img = img.rotate(180, expand=True)
        elif exif[ORIENTATION_KEY] == 6:
            img = img.rotate(270, expand=True)
        elif exif[ORIENTATION_KEY] == 8:
            img = img.rotate(90, expand=True)
    # Resize
    if img.size != INPUT_IMG_SIZE:
        # Crop to aspect ratio
        if img.size[0] / img.size[1] != INPUT_IMG_RATIO:
            if img.size[0] < img.size[1]:
                new_h = img.size[0] * 1 / INPUT_IMG_RATIO
                img = img.crop((0, 0, img.size[0], new_h))
            else:
                new_w = img.size[1] * INPUT_IMG_RATIO
                img = img.crop((0, 0, new_w, img.size[1]))

        resized = img.resize(INPUT_IMG_SIZE)
        resized.save(path)

    processed_path = os.path.join(RESULTS_FOLDER, os.path.basename(image_path))
    img.save(processed_path)
    
    # Run inference
    result = model(path, conf=DETECTION_THRESHOLD)[0]

    confs = result.boxes.conf
    clss = result.boxes.cls
    bboxs = result.boxes.xyxy

    draw = ImageDraw.Draw(img)

    sum_val = 0
    for j in range(len(confs)):
        box = list(bboxs[j])
        conf = float(confs[j])
        cls = int(clss[j])

        if conf < DETECTION_CONF_THRESHOLD:
            continue

        skip = False
        for k in range(len(confs)):
            if k == j:
                continue

            box2 = list(bboxs[k])
            conf2 = float(confs[k])
            iou = bb_intersection_over_union(box, box2)
            if iou >= IDENTICAL_IOU_THRESHOLD:
                if conf2 > conf:
                    skip = True
                    break
        if skip:
            continue

        sum_val += COIN_VALUES[cls]

        cls_name = CLASS_NAMES[cls]
        cls_color = COLOURS[cls]
        draw.rectangle(xy=box, outline=cls_color, width=4)
        det_label = cls_name + " - " + str(round(float(conf) * 100, 1)) + "%"
        draw.text(
            xy=(box[0], box[1] - 35),
            text=det_label,
            font=ImageFont.load_default(size=30),
            fill=cls_color,
        )
        
    result_path = os.path.join(RESULTS_FOLDER, f"result_{os.path.basename(path)}")
    img.save(result_path)


def clean_results():
    for f in os.listdir(RESULTS_FOLDER):
        os.remove(os.path.join(RESULTS_FOLDER, f))


if __name__ == "__main__":
    clean_results()

    images = glob.glob(os.path.join(UPLOAD_FOLDER, "*.jpg")) + glob.glob(os.path.join(UPLOAD_FOLDER, "*.png"))

    for img_path in images:
        process_image(img_path)
