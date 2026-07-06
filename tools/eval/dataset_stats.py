import os
import glob
import cv2
import tqdm
import argparse
import yaml
import numpy as np

try:
    from prettytable import PrettyTable
    HAS_PRETTYTABLE = True
except ImportError:
    HAS_PRETTYTABLE = False

COLOR_LIST = [
    (255, 0, 0),         # Red
    (0, 255, 0),         # Green
    (0, 0, 255),         # Blue
    (255, 165, 0),       # Orange
    (255, 255, 0),       # Yellow
    (0, 255, 255),       # Cyan
    (255, 0, 255),       # Magenta
    (255, 255, 255),     # White
    (128, 0, 0),         # Brown
    (0, 128, 0),         # Dark Green
    (0, 0, 128),         # Dark Blue
    (128, 128, 0),       # Olive
    (0, 128, 128),       # Teal
    (128, 0, 128),       # Purple
]

def get_color_by_class(class_id):
    return COLOR_LIST[class_id % len(COLOR_LIST)]

def draw_detections(box, name, color, img):
    height, width, _ = img.shape
    xmin, ymin, xmax, ymax = list(map(int, list(box)))
    line_thickness = max(1, int(min(height, width) / 400))
    font_scale = min(height, width) / 1000
    font_thickness = max(1, int(min(height, width) / 400))
    text_offset_y = int(min(height, width) / 100)
    
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, line_thickness)
    cv2.putText(img, str(name), (xmin, ymin - text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, lineType=cv2.LINE_AA)
    return img

def get_images_and_labels_path(images_folder_path, labels_folder_path):
    labels_filename = {}
    glob_list = glob.glob(os.path.join(labels_folder_path, '*.txt'))
    for i in glob_list:
        labels_filename[os.path.splitext(os.path.basename(i))[0]] = i
        
    image_postfix = ['jpg', 'png', 'bmp', 'tif', 'jpeg', 'webp']
    images_filename = {}
    for p in image_postfix:
        glob_list = glob.glob(os.path.join(images_folder_path, f'*.{p}'))
        for i in glob_list:
            images_filename[os.path.splitext(os.path.basename(i))[0]] = i
            
    print(f'Total images found: {len(images_filename)} | Total labels found: {len(labels_filename)}')
    
    image_label_dict = {}
    for name in labels_filename:
        if name in images_filename:
            image_label_dict[labels_filename[name]] = images_filename[name]
            
    print(f'Matched data pairs: {len(image_label_dict)}')
    return image_label_dict

def show_dataset_info(image_label_dict, classes, object_info, visual_box=False, save_path='runs/dataset_visual'):
    if visual_box:
        os.makedirs(save_path, exist_ok=True)

    classes_dict = {cls: {'s': 0, 'm': 0, 'l': 0, 'num': 0} for cls in classes}
    
    for label_path in tqdm.tqdm(image_label_dict, desc='Analyzing dataset'):
        image_path = image_label_dict[label_path]
        image = cv2.imread(image_path)
        if image is None:
            continue
            
        h, w = image.shape[:2]
        
        with open(label_path) as f:
            lines = f.readlines()
            label = [x.strip().split() for x in lines if x.strip()]
            
        for item in label:
            if len(item) < 5:
                continue
            cls_id, x_c, y_c, width, height = item[:5]
            cls_idx = int(float(cls_id))
            if cls_idx >= len(classes):
                continue
                
            cls_name = classes[cls_idx]
            classes_dict[cls_name]['num'] += 1
            
            width_val = float(width) * w
            height_val = float(height) * h
            obj_area = width_val * height_val
            
            if obj_area < object_info[0]:
                classes_dict[cls_name]['s'] += 1
            elif obj_area > object_info[1]:
                classes_dict[cls_name]['l'] += 1
            else:
                classes_dict[cls_name]['m'] += 1
                
            if visual_box:
                x_c_val, y_c_val = float(x_c) * w, float(y_c) * h
                x_min = x_c_val - width_val / 2
                y_min = y_c_val - height_val / 2
                x_max = x_c_val + width_val / 2
                y_max = y_c_val + height_val / 2
                image = draw_detections([x_min, y_min, x_max, y_max], cls_name, get_color_by_class(cls_idx), image)
                
        if visual_box:
            cv2.imwrite(os.path.join(save_path, os.path.basename(image_path)), image)

    total_s = sum(v['s'] for v in classes_dict.values())
    total_m = sum(v['m'] for v in classes_dict.values())
    total_l = sum(v['l'] for v in classes_dict.values())
    total_num = sum(v['num'] for v in classes_dict.values())
    
    if total_num == 0:
        print("No annotations found in dataset.")
        return

    if HAS_PRETTYTABLE:
        table = PrettyTable()
        table.field_names = ["Category", "Small (s)", "Medium (m)", "Large (l)", "Total (num)"]
        
        for category, values in classes_dict.items():
            s, m, l, num = values['s'], values['m'], values['l'], values['num']
            if num == 0:
                continue
            row = [
                category,
                f"{s} ({s/num:.1%})",
                f"{m} ({m/num:.1%})",
                f"{l} ({l/num:.1%})",
                num
            ]
            table.add_row(row)
            
        row_total = [
            "All",
            f"{total_s} ({total_s/total_num:.1%})",
            f"{total_m} ({total_m/total_num:.1%})",
            f"{total_l} ({total_l/total_num:.1%})",
            total_num
        ]
        table.add_row(row_total)
        table.align["Category"] = "l"
        print(table)
    else:
        # Text fallback if prettytable is not installed
        header = f"{'Category':<15} | {'Small (s)':<15} | {'Medium (m)':<15} | {'Large (l)':<15} | {'Total':<8}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for category, values in classes_dict.items():
            s, m, l, num = values['s'], values['m'], values['l'], values['num']
            if num == 0:
                continue
            print(f"{category:<15} | {s:<5} ({s/num:5.1%}) | {m:<5} ({m/num:5.1%}) | {l:<5} ({l/num:5.1%}) | {num:<8}")
        print("-" * len(header))
        print(f"{'All':<15} | {total_s:<5} ({total_s/total_num:5.1%}) | {total_m:<5} ({total_m/total_num:5.1%}) | {total_l:<5} ({total_l/total_num:5.1%}) | {total_num:<8}")
        print("-" * len(header))

def remap_yolo_dataset_class(image_label_dict, delete_label, output_labels_dir=None):
    labels_path_list = list(image_label_dict.keys())
    classes = []
    
    for label_path in tqdm.tqdm(labels_path_list, desc='Scanning label classes'):
        with open(label_path) as f:
            lines = f.readlines()
            label = [x.strip().split() for x in lines if x.strip()]
        for item in label:
            classes.append(int(float(item[0])))
            
    classes = sorted(list(set(classes)))
    filter_classes = list(sorted(set(classes) - set(delete_label)))
    print(f'Current dataset classes index: {classes}')
    print(f'Deleting indexes: {delete_label} | Remaining mapped classes indexes: {filter_classes}')
    
    if output_labels_dir:
        os.makedirs(output_labels_dir, exist_ok=True)

    for label_path in tqdm.tqdm(labels_path_list, desc='Processing remapping'):
        with open(label_path) as f:
            lines = f.readlines()
            label = [x.strip().split() for x in lines if x.strip()]
            
        new_label = []
        for item in label:
            cls_idx = int(float(item[0]))
            if cls_idx in delete_label:
                continue
            # Get new mapped index
            new_idx = filter_classes.index(cls_idx)
            new_label.append(' '.join([str(new_idx)] + item[1:5]))
            
        target_path = label_path
        if output_labels_dir:
            target_path = os.path.join(output_labels_dir, os.path.basename(label_path))
            
        with open(target_path, 'w+') as f:
            f.write('\n'.join(new_label))
            
    print(f"Remapping finished. Outputs written to: {output_labels_dir if output_labels_dir else 'Original path (in-place)'}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='data/images', help='images directory path')
    parser.add_argument('--labels', type=str, default='data/labels', help='labels directory path')
    parser.add_argument('--data-yaml', type=str, default='', help='dataset.yaml path')
    parser.add_argument('--classes', type=str, default='person,car', help='classes comma-separated if no yaml')
    parser.add_argument('--visualize', action='store_true', help='save images with bounding boxes')
    parser.add_argument('--save-path', type=str, default='runs/dataset_visual', help='save path for visualized images')
    parser.add_argument('--delete-label', type=str, default='', help='comma-separated class indexes to delete/filter out')
    parser.add_argument('--remap-output', type=str, default='', help='remap labels output directory (if empty, in-place remap)')
    parser.add_argument('--s-size', type=int, default=1024, help='small object area threshold (default: 32x32 = 1024)')
    parser.add_argument('--l-size', type=int, default=9216, help='large object area threshold (default: 96x96 = 9216)')
    
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
        
    image_label_dict = get_images_and_labels_path(args.images, args.labels)
    
    if len(image_label_dict) == 0:
        print("Error: No matching image-label pairs found. Check path names.")
        exit(1)
        
    object_info = [args.s-size if hasattr(args, 's-size') else args.s_size, 
                   args.l-size if hasattr(args, 'l-size') else args.l_size]
    
    # Show stats
    show_dataset_info(image_label_dict, class_names, object_info, visual_box=args.visualize, save_path=args.save_path)
    
    # Remap classes if delete-label is provided
    if args.delete_label:
        delete_indexes = [int(x) for x in args.delete_label.split(',')]
        remap_yolo_dataset_class(image_label_dict, delete_indexes, args.remap_output)
