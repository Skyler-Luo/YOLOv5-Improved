import os
import cv2
import tqdm
import shutil
import argparse
import yaml
import numpy as np

def xywh2xyxy(box):
    box_out = box.copy()
    box_out[:, 0] = box[:, 0] - box[:, 2] / 2
    box_out[:, 1] = box[:, 1] - box[:, 3] / 2
    box_out[:, 2] = box[:, 0] + box[:, 2] / 2
    box_out[:, 3] = box[:, 1] + box[:, 3] / 2
    return box_out

def iou(box1, box2):
    # box1: [N, 4], box2: [M, 4]
    x11, y11, x12, y12 = np.split(box1, 4, axis=1)
    x21, y21, x22, y22 = np.split(box2, 4, axis=1)
 
    xa = np.maximum(x11, np.transpose(x21))
    xb = np.minimum(x12, np.transpose(x22))
    ya = np.maximum(y11, np.transpose(y21))
    yb = np.minimum(y12, np.transpose(y22))
 
    area_inter = np.maximum(0, (xb - xa)) * np.maximum(0, (yb - ya))
 
    area_1 = (x12 - x11) * (y12 - y11)
    area_2 = (x22 - x21) * (y22 - y21)
    area_union = area_1 + np.transpose(area_2) - area_inter
 
    iou_matrix = area_inter / (area_union + 1e-8)
    return iou_matrix

def draw_box(img, box, color, label_text=""):
    height, width, _ = img.shape
    xmin, ymin, xmax, ymax = list(map(int, list(box)))
    line_thickness = max(1, int(min(height, width) / 300))
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness=line_thickness)
    if label_text:
        font_scale = min(height, width) / 800
        font_thickness = max(1, int(min(height, width) / 800))
        cv2.putText(img, label_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, lineType=cv2.LINE_AA)
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='data/images', help='ground truth images directory')
    parser.add_argument('--labels', type=str, default='data/labels', help='ground truth labels directory')
    parser.add_argument('--predictions', type=str, default='runs/predict', help='prediction label files directory')
    parser.add_argument('--output', type=str, default='runs/error_analysis', help='output directory for visualized images')
    parser.add_argument('--data-yaml', type=str, default='', help='dataset.yaml path')
    parser.add_argument('--classes', type=str, default='person,car', help='comma-separated class names if no yaml')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--postfix', type=str, default='jpg', help='image extension')
    
    args = parser.parse_args()
    
    # Extract classes
    if args.data_yaml:
        with open(args.data_yaml) as f:
            data = yaml.safe_load(f)
            class_names = data.get('names', [])
            if isinstance(class_names, dict):
                class_names = [class_names[i] for i in sorted(class_names.keys())]
    else:
        class_names = args.classes.split(',')
        
    # Visual colors: green = TP (detect), red = FN (missing), blue = FP (error/false alarm)
    detect_color = (0, 255, 0)
    missing_color = (0, 0, 255)
    error_color = (255, 0, 0)
    
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)

    all_right_num, all_missing_num, all_error_num = 0, 0, 0
    report_file = os.path.join(args.output, 'error_analysis_report.txt')
    
    label_files = [f for f in os.listdir(args.labels) if f.endswith('.txt')]
    print(f"Analyzing {len(label_files)} annotation files...")
    
    with open(report_file, 'w', encoding='utf-8') as f_w:
        for path in tqdm.tqdm(label_files):
            img_file_path = os.path.join(args.images, f'{path[:-4]}.{args.postfix}')
            image = cv2.imread(img_file_path)
            if image is None:
                print(f"Warning: Image file not found {img_file_path}", file=f_w)
                continue
                
            h, w = image.shape[:2]
            
            # Read predictions
            pred_file_path = os.path.join(args.predictions, path)
            pred = []
            if os.path.exists(pred_file_path):
                try:
                    with open(pred_file_path) as f:
                        lines = f.readlines()
                        if lines:
                            # Format: cls x_center y_center width height [conf]
                            temp_pred = np.array([list(map(float, x.strip().split())) for x in lines if x.strip()])
                            if temp_pred.ndim == 1:
                                temp_pred = np.expand_dims(temp_pred, axis=0)
                            if temp_pred.size > 0:
                                # Convert normalized xywh prediction to absolute xyxy
                                temp_pred[:, 1:5] = xywh2xyxy(temp_pred[:, 1:5])
                                temp_pred[:, [1, 3]] *= w
                                temp_pred[:, [2, 4]] *= h
                                pred = list(temp_pred)
                except Exception as e:
                    print(f"Error parsing prediction file {pred_file_path}: {e}", file=f_w)
            
            # Read ground truths
            label_file_path = os.path.join(args.labels, path)
            label = np.empty((0, 5))
            try:
                with open(label_file_path) as f:
                    lines = f.readlines()
                    if lines:
                        label = np.array([list(map(float, x.strip().split())) for x in lines if x.strip()])
                        if label.ndim == 1:
                            label = np.expand_dims(label, axis=0)
                        if label.size > 0:
                            label[:, 1:5] = xywh2xyxy(label[:, 1:5])
                            label[:, [1, 3]] *= w
                            label[:, [2, 4]] *= h
            except Exception as e:
                print(f"Error parsing label file {label_file_path}: {e}", file=f_w)
            
            right_num, missing_num, error_num = 0, 0, 0
            
            if label.size > 0:
                for i in range(label.shape[0]):
                    if len(pred) == 0:
                        # No predictions left, all remaining labels are missing (FN)
                        image = draw_box(image, label[i][1:5], missing_color, f"Missing {class_names[int(label[i, 0])]}")
                        missing_num += 1
                        continue
                        
                    # Calculate IoUs between current ground truth target and all predicted boxes
                    pred_boxes = np.array(pred)[:, 1:5]
                    ious = iou(label[i:i+1, 1:5], pred_boxes)[0]
                    ious_argsort = ious.argsort()[::-1]
                    
                    missing = True
                    for j in ious_argsort:
                        if ious[j] < args.iou_thres:
                            break
                        # Check if classes match
                        if int(label[i, 0]) == int(pred[j][0]):
                            cls_idx = int(pred[j][0])
                            cls_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
                            conf = pred[j][5] if len(pred[j]) > 5 else 1.0
                            image = draw_box(image, pred[j][1:5], detect_color, f"{cls_name} {conf:.2f}")
                            pred.pop(j)
                            missing = False
                            right_num += 1
                            break
                    
                    if missing:
                        cls_idx = int(label[i, 0])
                        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
                        image = draw_box(image, label[i][1:5], missing_color, f"Missing {cls_name}")
                        missing_num += 1
            
            # Any remaining predictions are false positives (FP / error)
            if len(pred) > 0:
                for j in range(len(pred)):
                    cls_idx = int(pred[j][0])
                    cls_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
                    conf = pred[j][5] if len(pred[j]) > 5 else 1.0
                    image = draw_box(image, pred[j][1:5], error_color, f"FP {cls_name} {conf:.2f}")
                    error_num += 1
            
            all_right_num += right_num
            all_missing_num += missing_num
            all_error_num += error_num
            
            cv2.imwrite(os.path.join(args.output, f'{path[:-4]}.{args.postfix}'), image)
            print(f'image:{path[:-4]} | True Positives (TP):{right_num} | False Negatives (FN):{missing_num} | False Positives (FP):{error_num}', file=f_w)
            
        print(f"\nSummary results across entire dataset:", file=f_w)
        print(f"Total True Positives (TP - Correctly Detected): {all_right_num}", file=f_w)
        print(f"Total False Negatives (FN - Missing Objects): {all_missing_num}", file=f_w)
        print(f"Total False Positives (FP - False Alarms/Errors): {all_error_num}", file=f_w)
        
    print("\n--- Error Analysis Complete ---")
    print(f"TP (Correct): {all_right_num}")
    print(f"FN (Missing): {all_missing_num}")
    print(f"FP (False alarms): {all_error_num}")
    print(f"Visualized files and summary report saved to {args.output}")
