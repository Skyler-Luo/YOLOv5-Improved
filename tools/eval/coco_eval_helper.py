import os
import cv2
import json
import argparse
import tqdm
import yaml

def yolo_to_coco(images_dir, labels_dir, classes_list, output_json_path):
    print(f"Converting YOLO format in {images_dir} to COCO format...")
    assert os.path.exists(images_dir), f"Images directory {images_dir} does not exist!"
    assert os.path.exists(labels_dir), f"Labels directory {labels_dir} does not exist!"
    
    image_postfix = ('.jpg', '.png', '.bmp', '.tif', '.jpeg', '.webp')
    image_names = [f for f in os.listdir(images_dir) if f.lower().endswith(image_postfix)]
    
    dataset = {'categories': [], 'annotations': [], 'images': []}
    for i, cls in enumerate(classes_list):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
        
    ann_id_cnt = 0
    for k, img_name in enumerate(tqdm.tqdm(image_names, desc="Processing images")):
        file_heads, _ = os.path.splitext(img_name)
        txt_file = f"{file_heads}.txt"
        
        img_path = os.path.join(images_dir, img_name)
        im = cv2.imread(img_path)
        if im is None:
            continue
        height, width, _ = im.shape
        
        label_path = os.path.join(labels_dir, txt_file)
        if not os.path.exists(label_path):
            continue
            
        # image_id can be integer if numeric, or string
        image_id = int(file_heads) if file_heads.isdigit() else file_heads
        
        dataset['images'].append({
            'file_name': img_name,
            'id': image_id,
            'width': width,
            'height': height
        })
        
        with open(label_path, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                
                # Convert normalized xywh to absolute xywh (x1, y1, width, height)
                x1 = (x - w / 2) * width
                y1 = (y - h / 2) * height
                box_width = w * width
                box_height = h * height
                
                dataset['annotations'].append({
                    'area': box_width * box_height,
                    'bbox': [x1, y1, box_width, box_height],
                    'category_id': cls_id,
                    'id': ann_id_cnt,
                    'image_id': image_id,
                    'iscrowd': 0,
                    'segmentation': [[x1, y1, x1 + box_width, y1, x1 + box_width, y1 + box_height, x1, y1 + box_height]]
                })
                ann_id_cnt += 1
                
    with open(output_json_path, 'w') as f:
        json.dump(dataset, f)
    print(f"Successfully converted. COCO JSON annotation saved to: {output_json_path}")

def run_coco_evaluation(gt_json, pred_json):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("Error: pycocotools is required for evaluation. Install it with: pip install pycocotools")
        return
        
    print(f"Loading ground truth JSON: {gt_json} ...")
    coco_gt = COCO(gt_json)
    print(f"Loading prediction JSON: {pred_json} ...")
    coco_dt = coco_gt.loadRes(pred_json)
    
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--convert', action='store_true', help='convert YOLO format dataset to COCO json')
    parser.add_argument('--eval', action='store_true', help='run COCO evaluation using ground truth and predictions JSON')
    parser.add_argument('--images', type=str, default='data/images/test', help='YOLO images directory')
    parser.add_argument('--labels', type=str, default='data/labels/test', help='YOLO labels directory')
    parser.add_argument('--data-yaml', type=str, default='', help='YOLO dataset.yaml path')
    parser.add_argument('--classes-txt', type=str, default='', help='classes.txt path')
    parser.add_argument('--classes', type=str, default='person,car', help='comma-separated class names if no files')
    parser.add_argument('--output', type=str, default='runs/coco_annotations.json', help='output JSON path for conversion')
    parser.add_argument('--gt', type=str, default='runs/coco_annotations.json', help='ground truth json for evaluation')
    parser.add_argument('--pred', type=str, default='runs/val/predictions.json', help='predictions json for evaluation')
    
    args = parser.parse_args()
    
    if args.convert:
        class_names = []
        if args.data_yaml:
            with open(args.data_yaml) as f:
                data = yaml.safe_load(f)
                class_names = data.get('names', [])
                if isinstance(class_names, dict):
                    class_names = [class_names[i] for i in sorted(class_names.keys())]
        elif args.classes_txt:
            with open(args.classes_txt) as f:
                class_names = [x.strip() for x in f.readlines() if x.strip()]
        else:
            class_names = args.classes.split(',')
            
        yolo_to_coco(args.images, args.labels, class_names, args.output)
        
    elif args.eval:
        run_coco_evaluation(args.gt, args.pred)
    else:
        parser.print_help()
