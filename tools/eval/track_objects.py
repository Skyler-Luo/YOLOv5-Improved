import warnings
warnings.filterwarnings('ignore')
import cv2
import os
import shutil
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO

try:
    from boxmot import DeepOCSORT, BYTETracker, BoTSORT, StrongSORT, OCSORT, HybridSORT
    BOXMOT_AVAILABLE = True
except ImportError:
    BOXMOT_AVAILABLE = False

def get_video_cfg(path):
    video = cv2.VideoCapture(path)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    return cv2.VideoWriter_fourcc(*'XVID'), size, fps

def counting(image_plot, result):
    box_count = result.boxes.shape[0]
    cv2.putText(image_plot, f'Object Counts: {box_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    return image_plot

def transform_mot(result):
    mot_result = []
    for i in range(result.boxes.shape[0]):
        mot_result.append(result.boxes.xyxy[i].cpu().detach().cpu().numpy().tolist() + [float(result.boxes.conf[i]), float(result.boxes.cls[i])])
    return np.array(mot_result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='YOLO weights path (.pt)')
    parser.add_argument('--source', type=str, default='data/video.mp4', help='input video file')
    parser.add_argument('--reid-weights', type=str, default='osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pt', help='ReID weights path')
    parser.add_argument('--tracker', type=str, default='deepocsort', choices=['deepocsort', 'botsort', 'strongsort', 'bytetracker', 'ocsort'], help='MOT tracker name')
    parser.add_argument('--output', type=str, default='runs/track', help='output folder')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or cpu')
    
    args = parser.parse_args()
    
    if not BOXMOT_AVAILABLE:
        print("Error: boxmot is not installed. Please run: pip install boxmot")
        exit(1)
        
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)
    
    device_str = f"cuda:{args.device}" if args.device != 'cpu' else 'cpu'
    
    # Initialize Tracker
    print(f"Initializing {args.tracker} with ReID weights: {args.reid-weights}")
    if args.tracker == 'deepocsort':
        tracker = DeepOCSORT(model_weights=Path(args.reid_weights), device=device_str, fp16=False)
    elif args.tracker == 'botsort':
        tracker = BoTSORT(model_weights=Path(args.reid_weights), device=device_str, fp16=False)
    elif args.tracker == 'strongsort':
        tracker = StrongSORT(model_weights=Path(args.reid_weights), device=device_str, fp16=False)
    elif args.tracker == 'bytetracker':
        tracker = BYTETracker()
    elif args.tracker == 'ocsort':
        tracker = OCSORT()
        
    print(f"Loading YOLO model: {args.weights}")
    model = YOLO(args.weights)
    
    fourcc, size, fps = get_video_cfg(args.source)
    out_video_path = os.path.join(args.output, os.path.basename(args.source))
    video_output = cv2.VideoWriter(out_video_path, fourcc, fps, size)
    
    print(f"Tracking objects in: {args.source}")
    results = model.predict(source=args.source, stream=True, imgsz=args.imgsz, save=False)
    
    for result in results:
        image_plot = result.orig_img
        mot_input = transform_mot(result)
        try:
            tracker.update(mot_input, image_plot)
            tracker.plot_results(image_plot, show_trajectories=True)
        except Exception as e:
            pass
        counting(image_plot, result)
        video_output.write(image_plot)
        
    video_output.release()
    print(f"Saved tracked video: {out_video_path}")
