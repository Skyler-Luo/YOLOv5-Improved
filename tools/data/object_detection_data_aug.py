import warnings
warnings.filterwarnings('ignore')
import os
import shutil
import cv2
import tqdm
import argparse
import yaml
import numpy as np
import albumentations as A
from PIL import Image
from multiprocessing import Pool
from typing import Callable, List

# Define the default Albumentations Compose Strategy
def get_augmentation_strategy():
    return A.Compose([
        A.Compose([
            A.Affine(scale=[0.5, 1.5], translate_percent=[0.0, 0.3], rotate=[-360, 360], shear=[-45, 45], keep_ratio=True, p=0.5),
            A.BBoxSafeRandomCrop(erosion_rate=0.2, p=0.1),
            A.D4(p=0.1),
            A.ElasticTransform(p=0.1),
            A.Flip(p=0.1),
            A.GridDistortion(p=0.1),
            A.Perspective(p=0.1),
        ], p=1.0),
        
        A.Compose([
            A.GaussNoise(p=0.1),
            A.ISONoise(p=0.1),
            A.ImageCompression(quality_lower=50, quality_upper=100, p=0.1),
            A.RandomBrightnessContrast(p=0.1),
            A.RandomFog(p=0.1),
            A.RandomRain(p=0.1),
            A.RandomSnow(p=0.1),
            A.RandomShadow(p=0.1),
            A.RandomSunFlare(p=0.1),
            A.ToGray(p=0.1),
        ], p=1.0)
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1, label_fields=['class_labels']))

def draw_detections(box, name, img):
    height, width, _ = img.shape
    xmin, ymin, xmax, ymax = list(map(int, list(box)))
    line_thickness = max(1, int(min(height, width) / 200))
    font_scale = min(height, width) / 500
    font_thickness = max(1, int(min(height, width) / 200))
    text_offset_y = int(min(height, width) / 50)
    
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), line_thickness)
    cv2.putText(img, str(name), (xmin, ymin - text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness, lineType=cv2.LINE_AA)
    return img

def show_labels(images_base_path, labels_base_path, save_path, classes):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    
    # Process a subset (max 100 images) for validation check
    image_names = os.listdir(images_base_path)[:100]
    for images_name in image_names:
        file_heads, _ = os.path.splitext(images_name)
        images_path = os.path.join(images_base_path, images_name)
        labels_path = os.path.join(labels_base_path, f'{file_heads}.txt')
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                lines = f.readlines()
                if not lines:
                    continue
                labels = np.array(list(map(lambda x: np.array(x.strip().split(), dtype=np.float64), lines)), dtype=np.float64)
            images = cv2.imread(images_path)
            if images is None:
                continue
            height, width, _ = images.shape
            # If labels is 1D (e.g. single target), reshape to 2D
            if labels.ndim == 1:
                labels = np.expand_dims(labels, axis=0)
            for cls, x_center, y_center, w, h in labels:
                x_center *= width
                y_center *= height
                w *= width
                h *= height
                class_idx = int(cls)
                class_name = classes[class_idx] if class_idx < len(classes) else str(class_idx)
                draw_detections([x_center - w // 2, y_center - h // 2, x_center + w // 2, y_center + h // 2], class_name, images)
            cv2.imwrite(os.path.join(save_path, images_name), images)

# Global variables for worker processes
_IMAGE_PATH = ""
_LABEL_PATH = ""
_AUG_IMAGE_PATH = ""
_AUG_LABEL_PATH = ""
_ENHANCEMENT_LOOP = 1
_STRATEGY = None

def init_worker(image_path, label_path, aug_image_path, aug_label_path, enhancement_loop):
    global _IMAGE_PATH, _LABEL_PATH, _AUG_IMAGE_PATH, _AUG_LABEL_PATH, _ENHANCEMENT_LOOP, _STRATEGY
    _IMAGE_PATH = image_path
    _LABEL_PATH = label_path
    _AUG_IMAGE_PATH = aug_image_path
    _AUG_LABEL_PATH = aug_label_path
    _ENHANCEMENT_LOOP = enhancement_loop
    _STRATEGY = get_augmentation_strategy()

def data_aug_single(images_name):
    try:
        file_heads, postfix = os.path.splitext(images_name)
        images_path = os.path.join(_IMAGE_PATH, images_name)
        labels_path = os.path.join(_LABEL_PATH, f'{file_heads}.txt')
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                lines = f.readlines()
                if not lines:
                    return
                labels = np.array(list(map(lambda x: np.array(x.strip().split(), dtype=np.float64), lines)), dtype=np.float64)
            
            # Read image in RGB format using PIL
            images = Image.open(images_path).convert('RGB')
            
            if labels.ndim == 1:
                labels = np.expand_dims(labels, axis=0)
                
            for i in range(_ENHANCEMENT_LOOP):
                new_images_name = os.path.join(_AUG_IMAGE_PATH, f'{file_heads}_aug_{i:0>3}{postfix}')
                new_labels_name = os.path.join(_AUG_LABEL_PATH, f'{file_heads}_aug_{i:0>3}.txt')
                
                # Make sure coordinates are within bounds [0, 1]
                bboxes = np.minimum(np.maximum(labels[:, 1:], 0.0), 1.0)
                
                transformed = _STRATEGY(
                    image=np.array(images), 
                    bboxes=bboxes, 
                    class_labels=labels[:, 0]
                )
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_class_labels = transformed['class_labels']
                
                # Save as BGR for OpenCV
                cv2.imwrite(new_images_name, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
                with open(new_labels_name, 'w+') as f_out:
                    for bbox, cls in zip(transformed_bboxes, transformed_class_labels):
                        f_out.write(f'{int(cls)} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n')
    except Exception as e:
        # Ignore errors on corrupted files
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='data/images/train', help='input images directory')
    parser.add_argument('--labels', type=str, default='data/labels/train', help='input labels directory')
    parser.add_argument('--output-images', type=str, default='data/images/train_aug', help='output images directory')
    parser.add_argument('--output-labels', type=str, default='data/labels/train_aug', help='output labels directory')
    parser.add_argument('--data-yaml', type=str, default='', help='dataset.yaml path to extract class names')
    parser.add_argument('--classes', type=str, default='person,car', help='comma-separated class names if no yaml')
    parser.add_argument('--loop', type=int, default=1, help='number of augmentations per image')
    parser.add_argument('--check-path', type=str, default='runs/aug_check', help='directory to draw labels on subset for checking')
    parser.add_argument('--workers', type=int, default=4, help='number of parallel workers')
    
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
        
    print(f"Loaded class names: {class_names}")
    
    if os.path.exists(args.output_images):
        shutil.rmtree(args.output_images)
    if os.path.exists(args.output_labels):
        shutil.rmtree(args.output_labels)
        
    os.makedirs(args.output_images, exist_ok=True)
    os.makedirs(args.output_labels, exist_ok=True)
    
    image_list = [f for f in os.listdir(args.images) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
    print(f"Found {len(image_list)} images. Starting parallel offline augmentation...")
    
    # Execute with Multiprocessing Pool
    with Pool(processes=args.workers, initializer=init_worker, 
              initargs=(args.images, args.labels, args.output_images, args.output_labels, args.loop)) as pool:
        list(tqdm.tqdm(pool.imap(data_aug_single, image_list), total=len(image_list)))
        
    print("Augmentation finished. Drawing visual check samples...")
    show_labels(args.output_images, args.output_labels, args.check_path, class_names)
    print(f"Visual check files saved to {args.check_path}")
