from ultralytics import YOLO
from ultralytics.data import utils

utils.IMG_FORMATS.add('ppm') 

model = YOLO(r"C:\Users\lavah\Desktop\ML\src\models\yolo\weights\best.pt")


model.predict(
    source=r"C:\Users\lavah\Desktop\ML\src\dataset\val\images",
    save=True,           
    conf=0.20            
)