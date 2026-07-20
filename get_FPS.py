"""
Inference Speed Benchmark Script
================================

Measures end-to-end inference throughput of the SAR ship detector on a
single device (GPU or CPU). The pipeline matches the speed reported in the
paper (104.2 FPS / 9.60 ms per image at 640x640 on a single GPU):

  preprocess (resize + pad + to_tensor) -> inference -> postprocess (NMS)

The script runs a warm-up pass followed by N timed iterations and reports
mean / std / total FPS. Use --device cpu to benchmark the CPU path and
--device 0 for a specific GPU.

Usage
-----
python get_FPS.py --weight runs/train/sar_ship_exp/weights/best.pt \
                  --imgsz 640 --iters 300 --warmup 20 --device 0
"""

import argparse
import time

import numpy as np


def benchmark(weight: str, imgsz: int, iters: int, warmup: int,
              device: str, batch: int, verbose: bool):
    from ultralytics import YOLO

    model = YOLO(weight)
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    print(f'[fps] Warming up ({warmup} iters) on device={device} ...')
    for _ in range(warmup):
        model.predict(source=dummy, imgsz=imgsz, device=device,
                      verbose=False, conf=0.25, save=False)

    timings = {'preprocess': [], 'inference': [], 'postprocess': [], 'total': []}
    print(f'[fps] Running {iters} timed iterations (batch={batch}) ...')
    for _ in range(iters):
        t0 = time.perf_counter()
        res = model.predict(source=[dummy] * batch, imgsz=imgsz, device=device,
                           verbose=False, conf=0.25, save=False)
        t1 = time.perf_counter()

        speed = res[0].speed
        timings['preprocess'].append(speed['preprocess'])
        timings['inference'].append(speed['inference'])
        timings['postprocess'].append(speed['postprocess'])
        timings['total'].append((t1 - t0) * 1000 / batch)

    summary = {}
    for k, vals in timings.items():
        arr = np.array(vals)
        summary[k] = (arr.mean(), arr.std())

    print('\n================ Throughput Summary ================')
    print(f'  weight     : {weight}')
    print(f'  imgsz      : {imgsz}x{imgsz}')
    print(f'  batch      : {batch}')
    print(f'  device     : {device}')
    print(f'  iters      : {iters} (warmup={warmup})')
    for k in ('preprocess', 'inference', 'postprocess', 'total'):
        m, s = summary[k]
        print(f'  {k:<11}: {m:7.2f} ms  (+/- {s:5.2f})')
    total_ms = summary['total'][0]
    print('----------------------------------------------------')
    print(f'  FPS (total) : {1000.0 / total_ms:7.2f}')
    print(f'  FPS (infer) : {1000.0 / summary["inference"][0]:7.2f}')
    print('====================================================\n')

    if verbose:
        import torch
        from ultralytics.utils.torch_utils import model_info
        try:
            n_l, n_p, n_g, flops = model_info(model.model)
            print(f'[fps] layers={n_l}, params={n_p/1e6:.2f}M, '
                  f'GFLOPs={n_g:.1f}, flops@{imgsz}={flops:.1f}')
        except Exception as e:
            print(f'[fps] model_info unavailable: {e}')

    return summary


def main():
    parser = argparse.ArgumentParser(description='Benchmark SAR ship detector throughput')
    parser.add_argument('--weight', type=str,
                        default='runs/train/sar_ship_exp/weights/best.pt')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--iters', type=int, default=300,
                        help='Timed iterations for FPS measurement')
    parser.add_argument('--warmup', type=int, default=20,
                        help='Warm-up iterations excluded from timing')
    parser.add_argument('--device', type=str, default='0',
                        help='cuda device (e.g. 0) or "cpu"')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--verbose', action='store_true', help='Print model info')
    args = parser.parse_args()

    benchmark(args.weight, args.imgsz, args.iters, args.warmup,
              args.device, args.batch, args.verbose)


if __name__ == '__main__':
    main()
