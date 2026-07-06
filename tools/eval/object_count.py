import warnings
warnings.filterwarnings('ignore')
import cv2
import os
import shutil
import argparse
from ultralytics import YOLO

def get_video_cfg(path):
    video = cv2.VideoCapture(path)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    return cv2.VideoWriter_fourcc(*'XVID'), size, fps

def plot_and_counting(result):
    image_plot = result.plot()
    box_count = result.boxes.shape[0]
    cv2.putText(image_plot, f'Object Counts: {box_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    return image_plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='YOLO model weights path (.pt)')
    parser.add_argument('--source', type=str, default='data/images', help='input images or video path')
    parser.add_argument('--output', type=str, default='runs/counting', help='output directory')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf', type=float, default=0.25, help='object confidence threshold')
    
    args = parser.parse_args()
    
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading Ultralytics YOLO model: {args.weights}")
    model = YOLO(args.weights)
    
    is_video = args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    if not is_video:
        print(f"Processing image source: {args.source}")
        results = model.predict(source=args.source, stream=True, imgsz=args.imgsz, conf=args.conf, save=False)
        for result in results:
            image_plot = plot_and_counting(result)
            out_file = os.path.join(args.output, os.path.basename(result.path))
            cv2.imwrite(out_file, image_plot)
            print(f"Saved counting result: {out_file}")
    else:
        print(f"Processing video source: {args.source}")
        video = cv2.VideoCapture(args.source)
        fourcc, size, fps = get_video_cfg(args.source)
        out_video_path = os.path.join(args.output, os.path.basename(args.source))
        out_video = cv2.VideoWriter(out_video_path, fourcc, fps, size)
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
            # Predict single frame
            results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, save=False)
            for result in results:
                image_plot = plot_and_counting(result)
                out_video.write(image_plot)
                
        video.release()
        out_video.release()
        print(f"Saved counting video: {out_video_path}")
