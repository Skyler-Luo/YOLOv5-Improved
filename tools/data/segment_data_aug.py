import warnings
warnings.filterwarnings('ignore')
import os
import shutil
import cv2
import tqdm
import argparse
import numpy as np
import albumentations as A
from PIL import Image
from multiprocessing import Pool

def generate_color_map(num_classes):
    hsv_colors = [(i * 180 // num_classes, 255, 255) for i in range(num_classes)]
    rgb_colors = [[0, 0, 0]] + [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2BGR)[0][0] for color in hsv_colors]
    return np.array(rgb_colors, dtype=np.uint8)

def get_augmentation_strategy():
    return A.Compose([
        A.Compose([
            A.Affine(scale=[0.5, 1.5], translate_percent=[0.0, 0.3], rotate=[-360, 360], shear=[-45, 45], keep_ratio=True, cval_mask=0, p=0.5),
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
    ], is_check_shapes=False)

def draw_segments(image, mask, colors):
    blended_image = cv2.addWeighted(image, 0.7, colors[mask], 0.3, 0)
    return blended_image

def show_labels(images_base_path, labels_base_path, save_path, colors):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    
    image_names = os.listdir(images_base_path)[:100]
    for images_name in image_names:
        file_heads, _ = os.path.splitext(images_name)
        images_path = os.path.join(images_base_path, images_name)
        labels_path = os.path.join(labels_base_path, f'{file_heads}.png')
        if os.path.exists(labels_path):
            images = cv2.imread(images_path)
            if images is None:
                continue
            masks = np.array(Image.open(labels_path))
            # Ensure mask colors fit
            if masks.max() >= len(colors):
                colors = generate_color_map(masks.max() + 5)
            images = draw_segments(images, masks, colors)
            cv2.imwrite(os.path.join(save_path, images_name), images)

# Global variables for workers
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
        labels_path = os.path.join(_LABEL_PATH, f'{file_heads}.png')
        if not os.path.exists(labels_path):
            labels_path = os.path.join(_LABEL_PATH, f'{file_heads}.jpg')
            
        if os.path.exists(labels_path):
            images = Image.open(images_path).convert('RGB')
            masks = Image.open(labels_path)
            
            for i in range(_ENHANCEMENT_LOOP):
                new_images_name = os.path.join(_AUG_IMAGE_PATH, f'{file_heads}_aug_{i:0>3}{postfix}')
                new_labels_name = os.path.join(_AUG_LABEL_PATH, f'{file_heads}_aug_{i:0>3}.png')
                
                transformed = _STRATEGY(
                    image=np.array(images), 
                    mask=np.array(masks)
                )
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']
                
                cv2.imwrite(new_images_name, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
                # Save mask as PNG to avoid compression loss
                Image.fromarray(transformed_mask).save(new_labels_name)
    except Exception as e:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='dataset/segment/images', help='input images folder')
    parser.add_argument('--masks', type=str, default='dataset/segment/labels', help='input masks folder (png/jpg)')
    parser.add_argument('--output-images', type=str, default='dataset/segment/images_aug', help='output images folder')
    parser.add_argument('--output-masks', type=str, default='dataset/segment/labels_aug', help='output masks folder')
    parser.add_argument('--loop', type=int, default=1, help='number of augmentations per image')
    parser.add_argument('--check-path', type=str, default='runs/seg_aug_check', help='check path')
    parser.add_argument('--workers', type=int, default=4, help='parallel workers')
    
    args = parser.parse_args()
    
    if os.path.exists(args.output_images):
        shutil.rmtree(args.output_images)
    if os.path.exists(args.output_masks):
        shutil.rmtree(args.output_masks)
        
    os.makedirs(args.output_images, exist_ok=True)
    os.makedirs(args.output_masks, exist_ok=True)
    
    image_list = [f for f in os.listdir(args.images) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
    print(f"Found {len(image_list)} images. Starting parallel offline semantic segmentation augmentation...")
    
    with Pool(processes=args.workers, initializer=init_worker, 
              initargs=(args.images, args.masks, args.output_images, args.output_masks, args.loop)) as pool:
        list(tqdm.tqdm(pool.imap(data_aug_single, image_list), total=len(image_list)))
        
    print("Augmentation finished. Drawing visual check samples...")
    colors = generate_color_map(80)
    show_labels(args.output_images, args.output_masks, args.check_path, colors)
    print(f"Visual check files saved to {args.check_path}")
