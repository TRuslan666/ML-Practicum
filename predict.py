from ultralytics import YOLO
from ultralytics.data import utils

utils.IMG_FORMATS.add('ppm') 

model = YOLO(r"C:\Users\lavah\Desktop\ML\runs\detect\results\plots\yolov8-2\weights\best.pt")


model.predict(
    source=r"C:\Users\lavah\Desktop\ML\data\processed\val\images",
    save=True,           
    conf=0.20            
)