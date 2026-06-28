#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SHAP Data Generator (YOLOv11 Classifier variant)
================================================
Same purpose as shap_generator.py but targets the YOLOv11Classifier model
defined in models.py instead of CNNModel1.

Output File Content (keys), identical to the CNN variant:
- 'shap_values': SHAP values in (C, H, W) format.
- 'image': Original normalized image in (C, H, W) format.
- 'prediction': int (Predicted class index)
- 'probabilities': array (Class probabilities)
"""

"""
EXAMPLE USAGE:
--------------

1. Prepare and Run (Complete Workflow):
CUDA_VISIBLE_DEVICES=0 python shap_generator_yolo.py prepare-and-run \
    --model_path "/path/to/your/yolo_checkpoint.pth" \
    --num_classes 2 --model_size "yolo11m" \
    --num_background 50 \
    --explainer_path "yolo_explainer.pkl" \
    --input_path "/path/to/images_folder" \
    --output_dir "./output_results" \
    --cuda_selection 0

2. Prepare Only:
python shap_generator_yolo.py prepare \
    --model_path "/path/to/yolo_checkpoint.pth" \
    --explainer_path "yolo_explainer.pkl"

3. Run Only:
python shap_generator_yolo.py run \
    --explainer_path "yolo_explainer.pkl" \
    --input_path "/path/to/single_image.png" \
    --output_dir "./output_results"
"""

import os
import time
import torch
import numpy as np
import cv2
import argparse
import pickle
from glob import glob
from models import YOLOv11Classifier
import warnings
import shap

warnings.filterwarnings('ignore')


def _format_duration(seconds):
    """Format seconds as Hh Mm Ss."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

# -----------------------------------------------------------------------------
# Data Saving Function
# -----------------------------------------------------------------------------

def save_shap_data(image_tensor, shap_values, prediction, probabilities, output_path):
    """
    Saves data in compressed numpy format (.npz).
    """
    if isinstance(image_tensor, torch.Tensor):
        img_np = image_tensor.cpu().detach().numpy()
        img_np = img_np[0]
    else:
        img_np = image_tensor

    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.cpu().detach().numpy()

    np.savez_compressed(
        output_path,
        shap_values=shap_values,
        image=img_np,
        prediction=prediction,
        probabilities=probabilities
    )

# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------

class ModelWrapper(torch.nn.Module):
    """
    Wraps the classifier so that SHAP receives softmax probabilities
    and a clean single-tensor forward signature.
    """
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        # YOLOv11Classifier.forward already unwraps tuple/list outputs,
        # but guard again here in case of upstream changes.
        if isinstance(output, (tuple, list)):
            output = output[0]
        return torch.nn.functional.softmax(output, dim=1)

def create_background_dataset(model, device, num_samples=50, img_size=300):
    print(f"Creating background dataset with {num_samples} samples...", flush=True)
    background_images = []
    for _ in range(num_samples):
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        img = img.astype(np.float32) / 255.0
        img = img.transpose((2, 0, 1))
        background_images.append(img)
    return torch.FloatTensor(np.array(background_images)).to(device)

def _load_yolo_state_dict(model, checkpoint):
    """
    Train pipeline saves checkpoints as {'model_state_dict': ..., 'optimizer_state_dict': ..., ...}
    (see train_deepscreen.py). Plain state_dicts are also supported for flexibility.
    """
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)

def prepare_explainer(model_path, num_classes, model_size, num_background,
                      cuda_selection, save_path, img_size=300):
    print("="*60); print("Mode: Preparing SHAP Generator (YOLOv11)"); print("="*60, flush=True)
    device = f'cuda:{cuda_selection}' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}", flush=True)

    try:
        model = YOLOv11Classifier(num_classes=num_classes, model_size=model_size).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        _load_yolo_state_dict(model, checkpoint)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}"); return None, None, None

    wrapped_model = ModelWrapper(model).to(device)
    background_images = create_background_dataset(model, device,
                                                  num_samples=num_background,
                                                  img_size=img_size)

    explainer = shap.GradientExplainer(wrapped_model, background_images)

    explainer_data = {
        'explainer': explainer,
        'model_path': model_path,
        'num_classes': num_classes,
        'model_size': model_size,
        'img_size': img_size,
        'device': device,
        'wrapped_model': wrapped_model,
    }

    with open(save_path, 'wb') as f:
        pickle.dump(explainer_data, f)
    return explainer, wrapped_model, device

def load_explainer(explainer_path):
    if not os.path.exists(explainer_path): return None, None, None, None
    with open(explainer_path, 'rb') as f:
        data = pickle.load(f)
    data['wrapped_model'].to(data['device'])
    data['wrapped_model'].eval()
    return data['explainer'], data['wrapped_model'], data['device'], data.get('img_size', 300)

def load_image(image_path, img_size=300):
    img = cv2.imread(image_path)
    if img is None: raise FileNotFoundError(f"Error: Could not load {image_path}")
    if img.shape[:2] != (img_size, img_size):
        img = cv2.resize(img, (img_size, img_size))

    img_normalized = img.astype(np.float32) / 255.0
    img_normalized = img_normalized.transpose((2, 0, 1))
    return torch.FloatTensor(img_normalized).unsqueeze(0)

def process_input_path(explainer, wrapped_model, device, input_path, output_dir,
                       nsamples=200, batch_size=200, img_size=300):
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    image_paths = []
    if os.path.isfile(input_path): image_paths.append(input_path)
    elif os.path.isdir(input_path):
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            image_paths.extend(glob(os.path.join(input_path, ext)))

    total = len(image_paths)
    print(f"Found {total} images. Starting generation...", flush=True)
    print(f"SHAP params: nsamples={nsamples}, batch_size={batch_size}, img_size={img_size}", flush=True)

    start_time = time.time()
    success = 0
    failed = 0

    for idx, image_path in enumerate(image_paths, start=1):
        iter_start = time.time()
        try:
            base_name = os.path.basename(image_path)
            file_name = os.path.splitext(base_name)[0]
            output_path = os.path.join(output_dir, f"{file_name}.npz")

            image_tensor = load_image(image_path, img_size=img_size).to(device)

            with torch.no_grad():
                output = wrapped_model(image_tensor)
                prediction = torch.argmax(output, dim=1).cpu().item()
                probabilities = output[0].cpu().numpy()

            image_tensor.requires_grad = True
            if not isinstance(explainer, shap.GradientExplainer):
                explainer = shap.GradientExplainer(wrapped_model, explainer.data)

            shap_values = explainer.shap_values(image_tensor, nsamples=nsamples)

            if isinstance(shap_values, list): shap_values = shap_values[prediction]
            if len(shap_values.shape) == 5: shap_values = shap_values[0, :, :, :, prediction]
            elif len(shap_values.shape) == 4 and shap_values.shape[0] == 1: shap_values = shap_values[0]

            save_shap_data(image_tensor, shap_values, prediction, probabilities, output_path)
            success += 1

            iter_time = time.time() - iter_start
            elapsed = time.time() - start_time
            avg = elapsed / idx
            remaining = avg * (total - idx)
            pct = 100.0 * idx / total
            print(
                f"[{idx}/{total}] ({pct:5.1f}%) {base_name} "
                f"| iter: {iter_time:.2f}s | avg: {avg:.2f}s "
                f"| elapsed: {_format_duration(elapsed)} "
                f"| eta: {_format_duration(remaining)}",
                flush=True,
            )

        except Exception as e:
            failed += 1
            print(f"[{idx}/{total}] FAILED {image_path}: {e}", flush=True)

    total_time = time.time() - start_time
    print("=" * 60, flush=True)
    print(
        f"Done. success={success}, failed={failed}, total={total} "
        f"| total_time={_format_duration(total_time)} "
        f"| avg={total_time / max(total, 1):.2f}s/img",
        flush=True,
    )
    print("=" * 60, flush=True)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SHAP Data Generator (YOLOv11)')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Common parameters for model setup
    model_p = argparse.ArgumentParser(add_help=False)
    model_p.add_argument('--model_path', type=str, required=True, help='Path to the .pth checkpoint file')
    model_p.add_argument('--num_classes', type=int, default=2, help='Number of output classes (default: 2)')
    model_p.add_argument('--model_size', type=str, default='yolo11m',
                         help='Ultralytics YOLO classifier size, e.g. yolo11n / yolo11s / yolo11m / yolo11l / yolo11x (default: yolo11m, matches train_deepscreen.py)')
    model_p.add_argument('--cuda_selection', type=int, default=0, help='GPU ID to use')
    model_p.add_argument('--num_background', type=int, default=50, help='Number of samples for SHAP background')
    model_p.add_argument('--explainer_path', type=str, default='shap_explainer_yolo.pkl', help='Path to save/load explainer')
    model_p.add_argument('--img_size', type=int, default=300, help='Input image size (default: 300, matches DEEPScreen pipeline)')

    # Common parameters for running analysis
    run_p = argparse.ArgumentParser(add_help=False)
    run_p.add_argument('--input_path', type=str, required=True, help='Path to image or directory of images')
    run_p.add_argument('--output_dir', type=str, required=True, help='Directory to save .npz results')

    # Dummy parameters to maintain compatibility with visualization scripts
    run_p.add_argument('--percentile', type=int, default=95, help='Ignored in generator mode')
    run_p.add_argument('--alpha', type=float, default=0.5, help='Ignored in generator mode')

    run_p.add_argument('--nsamples', type=int, default=200,
                       help='SHAP nsamples (default 200)')
    run_p.add_argument('--batch_size', type=int, default=200,
                       help='SHAP internal batch size for GPU utilization')

    subparsers.add_parser('prepare', parents=[model_p], help='Initialize and save the SHAP explainer')

    run_parser = subparsers.add_parser('run', parents=[run_p], help='Run generation using existing explainer')
    run_parser.add_argument('--explainer_path', type=str, required=True)

    subparsers.add_parser('prepare-and-run', parents=[model_p, run_p], help='Initialize explainer and run generation')

    args = parser.parse_args()

    if args.command == 'prepare':
        prepare_explainer(args.model_path, args.num_classes, args.model_size,
                          args.num_background, args.cuda_selection,
                          args.explainer_path, img_size=args.img_size)

    elif args.command == 'run':
        explainer, wrapped_model, device, img_size = load_explainer(args.explainer_path)
        if explainer:
            process_input_path(explainer, wrapped_model, device, args.input_path, args.output_dir,
                               nsamples=args.nsamples, batch_size=args.batch_size,
                               img_size=img_size)

    elif args.command == 'prepare-and-run':
        explainer, wrapped_model, device = prepare_explainer(
            args.model_path, args.num_classes, args.model_size,
            args.num_background, args.cuda_selection,
            args.explainer_path, img_size=args.img_size)
        if explainer:
            process_input_path(explainer, wrapped_model, device, args.input_path, args.output_dir,
                               nsamples=args.nsamples, batch_size=args.batch_size,
                               img_size=args.img_size)

if __name__ == '__main__':
    main()
