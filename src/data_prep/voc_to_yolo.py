"""Converts Pascal VOC XML annotations (as used by the US-Seville weapon
datasets) into YOLO-format .txt labels. Rejects malformed boxes rather than
silently writing bad labels."""

import argparse
import glob
import os
import random
import shutil
import xml.etree.ElementTree as ET

CLASS_MAP = {'Handgun': 0, 'Knife': 1, 'Rifle': 2, 'Short_rifle': 2}
CLASS_NAMES = ['Handgun', 'Knife', 'Rifle']


def convert_voc_to_yolo(xml_path):
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None
    root = tree.getroot()
    filename_elem = root.find('filename')
    if filename_elem is None:
        return None
    img_filename = filename_elem.text
    if not img_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_filename = os.path.splitext(os.path.basename(xml_path))[0] + '.jpg'
    size = root.find('size')
    if size is None:
        return None
    img_w, img_h = int(size.find('width').text), int(size.find('height').text)
    if img_w <= 0 or img_h <= 0:
        return None

    lines = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASS_MAP:
            continue
        class_id = CLASS_MAP[name]
        bbox = obj.find('bndbox')
        xmin, ymin = float(bbox.find('xmin').text), float(bbox.find('ymin').text)
        xmax, ymax = float(bbox.find('xmax').text), float(bbox.find('ymax').text)
        x_center, y_center = ((xmin + xmax) / 2) / img_w, ((ymin + ymax) / 2) / img_h
        box_w, box_h = (xmax - xmin) / img_w, (ymax - ymin) / img_h
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < box_w <= 1 and 0 < box_h <= 1):
            continue
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
    return img_filename, lines


def process_records(records, out_dir, split_name):
    converted = 0
    for xml_path, img_dir in records:
        result = convert_voc_to_yolo(xml_path)
        if result is None:
            continue
        img_filename, lines = result
        img_src = os.path.join(img_dir, img_filename)
        if not os.path.exists(img_src):
            continue
        base_name = os.path.splitext(img_filename)[0]
        unique_name = f"{os.path.basename(img_dir).replace(' ', '_')}_{base_name}"
        img_dst = os.path.join(out_dir, 'images', split_name, unique_name + '.jpg')
        label_dst = os.path.join(out_dir, 'labels', split_name, unique_name + '.txt')
        shutil.copy(img_src, img_dst)
        with open(label_dst, 'w') as f:
            f.write('\n'.join(lines))
        converted += 1
    return converted


def main(source_dirs, out_dir, val_fraction=0.1, seed=42):
    for split in ['train', 'val']:
        os.makedirs(os.path.join(out_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'labels', split), exist_ok=True)

    all_records = []
    for source_dir in source_dirs:
        for xml_path in glob.glob(os.path.join(source_dir, '*.xml')):
            all_records.append((xml_path, source_dir))

    random.seed(seed)
    random.shuffle(all_records)
    split_idx = int(len(all_records) * (1 - val_fraction))
    train_records, val_records = all_records[:split_idx], all_records[split_idx:]

    train_count = process_records(train_records, out_dir, 'train')
    val_count = process_records(val_records, out_dir, 'val')
    print(f"Converted: Train {train_count}, Val {val_count}")

    yaml_content = f"path: {out_dir}\ntrain: images/train\nval: images/val\n\nnc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n"
    with open(os.path.join(out_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    print(f"data.yaml written to {out_dir}/data.yaml")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dirs', nargs='+', required=True,
                         help='One or more directories containing VOC XML + image pairs')
    parser.add_argument('--out_dir', default='data/weapon_yolo_dataset')
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args.source_dirs, args.out_dir, args.val_fraction, args.seed)
