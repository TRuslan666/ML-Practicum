from ultralytics import YOLO
from ultralytics.data import utils


# Разрешаем YOLO читать формат .ppm
utils.IMG_FORMATS.add('ppm') 

def main():
    # Загружаем базовую модель
    model = YOLO(r"C:\Users\lavah\Desktop\ML-Project\src\models\yolo\weights\last.pt") 

    # Запускаем обучение
    model.train(
        data="C:\Users\lavah\Desktop\ML-Project\src\dataset\dataset.yaml", 
        epochs=50, 
        imgsz=640, 
        batch=16, 
        device=0,      
        amp=False,     
        cache=False,   
        workers=0      
    )

if __name__ == '__main__':
    main()