import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
import argparse
import tqdm

def convert_coordinates(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def convert_annotation(xml_path, img_dir, out_dir, img_name, postfix, classes_list, auto_classes):
    xml_file = os.path.join(xml_path, f"{img_name}.xml")
    if not os.path.exists(xml_file):
        return []
        
    with open(xml_file, "r", encoding='utf-8') as in_file:
        tree = ET.parse(in_file)
        root = tree.getroot()
        
        # Read image to get exact width/height (highly accurate)
        img_path = os.path.join(img_dir, f"{img_name}.{postfix}")
        img = cv2.imread(img_path)
        if img is None:
            # Fallback to size block in XML
            size_elem = root.find('size')
            if size_elem is not None:
                w = int(size_elem.find('width').text)
                h = int(size_elem.find('height').text)
            else:
                return []
        else:
            h, w = img.shape[:2]
            
        res = []
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes_list:
                if auto_classes:
                    classes_list.append(cls)
                else:
                    continue
            cls_id = classes_list.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert_coordinates((w, h), b)
            res.append(f"{cls_id} " + " ".join([f"{a:.6f}" for a in bb]))
            
        if len(res) > 0:
            txt_file = os.path.join(out_dir, f"{img_name}.txt")
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(res))
    return classes_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml-dir', type=str, default='VOCdevkit/Annotations', help='xml files directory')
    parser.add_argument('--img-dir', type=str, default='VOCdevkit/JPEGImages', help='images directory')
    parser.add_argument('--out-dir', type=str, default='VOCdevkit/labels', help='output labels directory')
    parser.add_argument('--classes-txt', type=str, default='', help='path to classes.txt if predefined')
    parser.add_argument('--classes', type=str, default='', help='comma-separated class names if predefined')
    parser.add_argument('--postfix', type=str, default='jpg', help='image files extension')
    
    args = parser.parse_args()
    
    # Setup classes
    classes = []
    auto_classes = True
    if args.classes_txt:
        if os.path.exists(args.classes_txt):
            with open(args.classes_txt) as f:
                classes = [x.strip() for x in f.readlines() if x.strip()]
            auto_classes = False
    elif args.classes:
        classes = args.classes.split(',')
        auto_classes = False
        
    os.makedirs(args.out_dir, exist_ok=True)
    
    xml_files = [f for f in os.listdir(args.xml_dir) if f.lower().endswith('.xml')]
    print(f"Starting conversion of {len(xml_files)} XML annotations...")
    
    error_list = []
    for xml_name in tqdm.tqdm(xml_files):
        img_name = os.path.splitext(xml_name)[0]
        try:
            classes = convert_annotation(
                args.xml_dir, args.img_dir, args.out_dir, img_name, args.postfix, classes, auto_classes
            )
        except Exception as e:
            error_list.append(xml_name)
            
    print(f"\nConversion finished.")
    if error_list:
        print(f"Failed files count: {len(error_list)} (details: {error_list})")
    print(f"Dataset classes detected: {classes}")
    
    # Save a classes.txt in output folder for record
    classes_file = os.path.join(args.out_dir, 'classes.txt')
    with open(classes_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(classes))
    print(f"Classes list written to: {classes_file}")
