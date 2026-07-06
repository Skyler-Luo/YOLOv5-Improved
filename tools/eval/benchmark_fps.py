import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import time
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils.torch_utils import select_device
from models.common import DetectMultiBackend

def get_weight_size(path):
    try:
        stats = os.stat(path)
        return f'{stats.st_size / 1024 / 1024:.1f}'
    except:
        return 'Unknown'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='trained PyTorch weights path (.pt or .pkl)')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--imgs', nargs='+', type=int, default=[640, 640], help='[height, width] image sizes')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--warmup', default=200, type=int, help='number of warmup iterations')
    parser.add_argument('--runs', default=1000, type=int, help='number of benchmark test iterations')
    parser.add_argument('--half', action='store_true', help='fp16 half-precision inference mode')
    
    args = parser.parse_args()
    
    device = select_device(args.device, batch_size=args.batch)
    
    # Check shape format
    if len(args.imgs) == 1:
        img_size = (args.imgs[0], args.imgs[0])
    else:
        img_size = (args.imgs[0], args.imgs[1])
        
    print(f"Loading weights: {args.weights} ...")
    model = DetectMultiBackend(args.weights, device=device)
    model.eval()
    
    example_inputs = torch.randn((args.batch, 3, *img_size)).to(device)
    
    if args.half:
        model = model.half()
        example_inputs = example_inputs.half()
    else:
        model = model.float()
        example_inputs = example_inputs.float()
        
    print(f"Starting GPU/CPU warmups ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        model(example_inputs)
        
    print(f"Measuring latency ({args.runs} iterations)...")
    time_arr = []
    
    for _ in tqdm(range(args.runs), desc='Inference benchmark'):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        
        model(example_inputs)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        time_arr.append(end_time - start_time)
        
    std_time = np.std(time_arr)
    mean_time = np.mean(time_arr)
    # Latency per image
    infer_time_per_image = mean_time / args.batch
    fps = 1.0 / infer_time_per_image
    
    print("\n" + "=" * 50)
    print("Benchmark Results Summary:")
    print("=" * 50)
    print(f"Weights File:       {args.weights}")
    print(f"Weights Size:       {get_weight_size(args.weights)} MB")
    print(f"Device:             {device}")
    print(f"Batch Size:         {args.batch}")
    print(f"Resolution (H x W): {img_size[0]} x {img_size[1]}")
    print(f"Precision:          {'FP16' if args.half else 'FP32'}")
    print(f"Mean Latency/Batch: {mean_time * 1000:.3f} ms")
    print(f"Mean Latency/Image: {infer_time_per_image * 1000:.3f} ms (± {std_time * 1000:.3f} ms)")
    print(f"Throughput (FPS):   {fps:.2f} frames/sec")
    print("=" * 50)
