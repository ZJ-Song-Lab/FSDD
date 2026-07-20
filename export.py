"""
Edge Deployment Export Script (Atlas 200I DK A2 / Ascend 310B)
==============================================================

Reproduces the deployment pipeline described in the paper:
  PyTorch (.pt) -> ONNX -> ATC-compiled .om (FP16) on Atlas 200I DK A2.

Steps
-----
1. Export the trained YOLOv8l-FSEM/MSFE/SOEP checkpoint to ONNX (opset 11,
   static 640x640, NMS built-in). Run this step on the training host.
2. (Optional) Simplify the ONNX graph with onnx-simplifier to remove
   redundant cast/shape ops before ATC compilation.
3. Convert ONNX -> .om with the Huawei ATC compiler shipped with CANN 8.0.
   The .om runs with FP16 compute on the Ascend 310B NPU.
4. (Optional) Post-training INT8 quantization via the AMCT toolkit using a
   small calibration split of the SAR dataset.

Usage
-----
# 1. PT -> ONNX (on the training host)
python export.py --weight runs/train/sar_ship_exp/weights/best.pt \
                 --imgsz 640 --simplify

# 2. ONNX -> OM (on the Atlas 200I DK A2, with CANN 8.0 + ATC in PATH)
python export.py --onnx runs/train/sar_ship_exp/weights/best.onnx \
                 --soc Ascend310B4 --fp16

# 3. (Optional) INT8 quantization with AMCT
python export.py --onnx runs/train/sar_ship_exp/weights/best.onnx \
                 --soc Ascend310B4 --int8 --calib datasets/SSDD/images/train
"""

import argparse
import os
import subprocess
import sys


def export_pt_to_onnx(weight: str, imgsz: int, simplify: bool, batch: int = 1):
    """Step 1: PyTorch checkpoint -> ONNX via ultralytics export API."""
    from ultralytics import YOLO

    model = YOLO(weight)
    onnx_path = model.export(
        format='onnx',
        imgsz=imgsz,
        batch=batch,
        opset=11,
        simplify=simplify,
        dynamic=False,
        half=False,
    )
    print(f'[export] ONNX saved to: {onnx_path}')
    return onnx_path


def simplify_onnx(onnx_path: str):
    """Step 2 (optional): onnx-simplifier to clean up the graph for ATC."""
    try:
        import onnxsim
        import onnx
    except ImportError:
        print('[export] onnxsim/onnx not installed, skipping simplification')
        return onnx_path

    sim_path = onnx_path.replace('.onnx', '_sim.onnx')
    model = onnx.load(onnx_path)
    simplified, check = onnxsim.simplify(model)
    if check:
        onnx.save(simplified, sim_path)
        print(f'[export] Simplified ONNX saved to: {sim_path}')
        return sim_path
    print('[export] Simplification failed validation, keeping original ONNX')
    return onnx_path


def atc_compile(onnx_path: str, soc: str, fp16: bool, int8: bool,
               calib: str = None, out_dir: str = 'runs/export'):
    """Step 3/4: ONNX -> .om via the ATC compiler (requires CANN toolkit)."""
    os.makedirs(out_dir, exist_ok=True)
    om_path = os.path.join(out_dir, os.path.splitext(os.path.basename(onnx_path))[0] + '.om')

    cmd = [
        'atc',
        f'--model={onnx_path}',
        f'--framework=5',                  # 5 = ONNX
        f'--output={os.path.splitext(om_path)[0]}',
        f'--soc_version={soc}',
        '--input_shape="images:{batch},3,{img},{img}"'.format(batch=1, img=640),
    ]
    if fp16:
        cmd.append('--precision_mode=force_fp16')
    elif int8:
        if calib is None:
            print('[export] INT8 quantization requires --calib image dir')
            sys.exit(1)
        cmd.append('--precision_mode=allow_mix_precision')
        cmd.append(f'--calibration_dataset_path={calib}')

    print('[export] Running ATC:\n  ' + ' '.join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print('[export] ATC not found. Run this step on the Atlas 200I DK A2 '
              'with CANN 8.0 environment sourced (set_env.sh).')
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f'[export] ATC failed: {e}')
        sys.exit(1)

    print(f'[export] OM model saved to: {om_path}')
    return om_path


def main():
    parser = argparse.ArgumentParser(description='Export SAR ship detector for edge deployment')
    parser.add_argument('--weight', type=str, default='runs/train/sar_ship_exp/weights/best.pt',
                        help='PyTorch checkpoint (.pt) for step 1')
    parser.add_argument('--onnx', type=str, default=None,
                        help='Existing ONNX path for step 2/3 (skip step 1)')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--batch', type=int, default=1, help='Static batch size for ONNX')
    parser.add_argument('--simplify', action='store_true', help='Run onnx-simplifier')
    parser.add_argument('--soc', type=str, default='Ascend310B4',
                        help='Ascend SoC version for ATC (Atlas 200I DK A2 = Ascend310B4)')
    g = parser.add_mutually_exclusive_group()
    g.add_argument('--fp16', action='store_true', help='Compile .om in FP16 (default)')
    g.add_argument('--int8', action='store_true', help='Compile .om in INT8 (PTQ)')
    parser.add_argument('--calib', type=str, default=None,
                        help='Calibration image dir for INT8 PTQ')
    args = parser.parse_args()

    fp16 = True if not args.int8 else False

    onnx_path = args.onnx
    if onnx_path is None:
        onnx_path = export_pt_to_onnx(args.weight, args.imgsz, args.simplify, args.batch)
        if args.simplify:
            onnx_path = simplify_onnx(onnx_path)
    else:
        print(f'[export] Using existing ONNX: {onnx_path}')

    atc_compile(onnx_path, args.soc, fp16, args.int8, args.calib)


if __name__ == '__main__':
    main()
