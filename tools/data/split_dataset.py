import os
import shutil
import random
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='VOCdevkit/JPEGImages', help='input images folder')
    parser.add_argument('--labels', type=str, default='VOCdevkit/labels', help='input labels folder')
    parser.add_argument('--output', type=str, default='dataset', help='output root dataset folder')
    parser.add_argument('--val-size', type=float, default=0.1, help='validation split ratio')
    parser.add_argument('--test-size', type=float, default=0.2, help='test split ratio')
    parser.add_argument('--postfix', type=str, default='jpg', help='image extension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # Define paths
    out_img_train = os.path.join(args.output, 'images/train')
    out_img_val = os.path.join(args.output, 'images/val')
    out_img_test = os.path.join(args.output, 'images/test')
    
    out_lbl_train = os.path.join(args.output, 'labels/train')
    out_lbl_val = os.path.join(args.output, 'labels/val')
    out_lbl_test = os.path.join(args.output, 'labels/test')
    
    os.makedirs(out_img_train, exist_ok=True)
    os.makedirs(out_img_val, exist_ok=True)
    os.makedirs(out_img_test, exist_ok=True)
    
    os.makedirs(out_lbl_train, exist_ok=True)
    os.makedirs(out_lbl_val, exist_ok=True)
    os.makedirs(out_lbl_test, exist_ok=True)
    
    label_files = [x for x in os.listdir(args.labels) if x.endswith('.txt') and x != 'classes.txt']
    random.shuffle(label_files)
    
    total_count = len(label_files)
    if total_count == 0:
        print(f"Error: No text label files found in {args.labels}")
        exit(1)
        
    val_count = int(total_count * args.val_size)
    test_count = int(total_count * args.test_size)
    train_count = total_count - val_count - test_count
    
    train_files = label_files[:train_count]
    val_files = label_files[train_count:train_count + val_count]
    test_files = label_files[train_count + val_count:]
    
    print(f"Dataset split config: Train: {train_count} | Val: {val_count} | Test: {test_count}")
    
    # Helper copy function
    def copy_pairs(files, img_dest, lbl_dest, subset_name):
        copied_cnt = 0
        for f in files:
            file_heads = os.path.splitext(f)[0]
            img_src = os.path.join(args.images, f"{file_heads}.{args.postfix}")
            lbl_src = os.path.join(args.labels, f)
            
            if os.path.exists(img_src) and os.path.exists(lbl_src):
                shutil.copy(img_src, os.path.join(img_dest, f"{file_heads}.{args.postfix}"))
                shutil.copy(lbl_src, os.path.join(lbl_dest, f))
                copied_cnt += 1
        print(f"Successfully copied {copied_cnt}/{len(files)} pairs to {subset_name}")
        
    copy_pairs(train_files, out_img_train, out_lbl_train, 'Train')
    copy_pairs(val_files, out_img_val, out_lbl_val, 'Val')
    copy_pairs(test_files, out_img_test, out_lbl_test, 'Test')
    
    # Copy classes.txt if present
    classes_src = os.path.join(args.labels, 'classes.txt')
    if os.path.exists(classes_src):
        shutil.copy(classes_src, os.path.join(args.output, 'classes.txt'))
        print("Copied classes.txt to dataset root folder.")
        
    print(f"\n--- Split Complete ---")
    print(f"Outputs written to: {args.output}")
