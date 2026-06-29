import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import cv2
import numpy as np

# 1. Настройка девайса
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Список ваших классов (всего 40 классов знаков + 1 фон)
# ВАЖНО: В PyTorch Faster R-CNN класс 0 ВСЕГДА зарезервирован под фон (background)
# Ваши классы знаков смещаются на +1. То есть:
# class_0 -> ID 1, class_1 -> ID 2 ... class_39 -> ID 40
categories = ["background"] + [f"sign_class_{i}" for i in range(43)] 
num_classes = len(categories) 

# 3. Инициализация кастомной модели
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

# Перестраиваем классификатор под 41 класс
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# 4. Загрузка ваших весов
checkpoint = torch.load('results/faster_rcnn/best.pt', map_location=device)
state_dict = checkpoint['model_state_dict'] # <--- Достаем чистые веса из чекпоинта
model.load_state_dict(state_dict)

# Перенос на девайс и режим инференса
model.to(device)
model.eval()

# 5. Загрузка и предобработка изображения
image_path = "test_image.jpg"  # Замените на путь к вашему изображению
img_pil = Image.open(image_path).convert("RGB")

# Преобразуем PIL Image в тензор PyTorch [C, H, W] и делим на 255
img_tensor = F.to_tensor(img_pil).to(device)

# Добавляем batch-размерность: [C, H, W] -> [1, C, H, W]
img_tensor = img_tensor.unsqueeze(0)

# 6. Запуск инференса
with torch.no_grad():
    predictions = model(img_tensor)

# 7. Обработка результатов
pred = predictions[0]

boxes = pred['boxes'].cpu().numpy()     # [[xmin, ymin, xmax, ymax], ...]
labels = pred['labels'].cpu().numpy()   # Массив ID классов (целые числа)
scores = pred['scores'].cpu().numpy()   # Массив вероятностей от 0.0 до 1.0

# 8. Визуализация с помощью OpenCV
img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

CONFIDENCE_THRESHOLD = 0.5  # Порог уверенности

for box, label, score in zip(boxes, labels, scores):
    if score > CONFIDENCE_THRESHOLD:
        xmin, ymin, xmax, ymax = map(int, box)
        
        # Получаем имя класса по его ID. Безопасно извлекаем, 
        # чтобы код не упал, если модель предскажет несуществующий индекс
        class_name = categories[label] if label < len(categories) else f"unknown_{label}"
        
        # Рисуем рамку
        cv2.rectangle(img_cv, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # Пишем текст (Класс: Уверенность)
        text = f"{class_name}: {score:.2f}"
        cv2.putText(img_cv, text, (xmin, ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Сохраняем результат
cv2.imwrite("result.jpg", img_cv)
print("Инференс завершен, результат сохранен в 'result.jpg'")