import warnings

warnings.filterwarnings('ignore')

"""
SAR Ship Detection Training Script
==================================

Baseline detector follows the YOLOv8l architecture with a bidirectional PAFPN
neck and an anchor-free decoupled head predicting at P3-P5. Three modules from
the paper are integrated on top of it:

1. FSEM  - Frequency-Spatial Enhancement Module  (Scharr edge + FFT dual-domain fusion)
2. MSFE  - Multi-Stage Feature Enhancement        (SAPA polarized linear attention + DyT + EDFFN)
3. SOEP  - Small-Object Enhance Pyramid            (SPDConv + OmniKernel global-local fusion)

Model configs (pick one):
  - FSDD/cfg/models/COMP.yaml  : full model (FSEM + MSFE + SOEP)
  - FSDD/cfg/models/SOEP.yaml  : ablation without FSEM
  - FSDD/cfg/models/MSFE.yaml  : ablation variant

Datasets (see datasets/*.yaml):
  - datasets/SSDD.yaml   (928 train / 232 test)
  - datasets/HRSID.yaml  (3642 train / 1962 test)
  - datasets/RSDD.yaml   (5000 train / 2000 test)

Paper training schedule (for reference): SGD, lr0=0.01, momentum=0.937,
weight_decay=5e-4, 3-epoch linear warmup -> cosine to 1e-4, batch=16,
600 epochs, mixed-precision, seed=42, EMA, imgsz=640, single-channel SAR
replicated to 3 channels, HSV hue/sat gains zeroed.
"""

from ultralytics import YOLO

if __name__ == '__main__':
    # Full model (FSEM + MSFE + SOEP). For ablations use SOEP.yaml or MSFE.yaml.
    model = YOLO('FSDD/cfg/models/COMP.yaml')

    model.train(
        data='datasets/SSDD.yaml',   # one of: SSDD.yaml / HRSID.yaml / RSDD.yaml
        cache=False,
        imgsz=640,
        epochs=600,                  # paper schedule; use a smaller value for a quick smoke test
        batch=16,
        lr0=0.01,                    # SGD initial learning rate (paper)
        momentum=0.937,
        weight_decay=5e-4,
        workers=4,
        seed=42,
        project='runs/train',
        name='sar_ship_exp',
    )

    """
    Example usage:

    # Train on a different dataset:
    model.train(data='datasets/HRSID.yaml', imgsz=640, epochs=600, batch=16)

    # Run an ablation config:
    model = YOLO('FSDD/cfg/models/SOEP.yaml')
    model.train(data='datasets/SSDD.yaml', epochs=600, lr0=0.01)
    """
