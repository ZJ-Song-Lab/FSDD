import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

"""
SAR Ship Detection Training Script

This script uses the optimized model configuration with three key enhancement modules:
1. Frequency-Spatial Enhancement Module (FSEM) - For noise suppression and structure preservation
2. Multi-Stage Feature Enhancement (MSFE) - For scattering-aware feature stabilization
3. Small-Object Enhance Pyramid (SOEP) - For efficient small-target recovery

These modules are specifically designed to improve SAR ship detection performance.
"""

if __name__ == '__main__':
    # Load the optimized model configuration with SAR-specific enhancements
    model = RTDETR('FSDD/cfg/models/COMP.yaml')
    
    # Train the model with SAR ship detection data
    model.train(data='path/to/sar_ship_data.yaml',  # Update with your actual data path
                cache=False,
                imgsz=640,
                epochs=100,
                batch=16,
                workers=4,
                project='runs/train',
                name='sar_ship_exp',
                )
    
    """
    Example usage:
    
    # To train with different configurations:
    # 1. Default optimized model (includes all three enhancement modules)
    model = RTDETR('FSDD/cfg/models/COMP.yaml')
    
    # 2. Train with specific hyperparameters
    model.train(
        data='path/to/sar_ship_data.yaml',
        imgsz=640,
        epochs=200,
        batch=8,
        lr0=0.001,
        project='runs/train',
        name='sar_ship_optimal'
    )
    """
