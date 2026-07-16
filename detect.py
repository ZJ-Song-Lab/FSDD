import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train/sar_ship_exp/weights/best.pt')
    model.predict(source='datasets/SSDD/images/test',
                  conf=0.25,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  )