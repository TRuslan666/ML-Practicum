import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw

# 1. Указываем путь к папке, куда всё сохранилось
model_path = "my_best_detr_model"

# 2. Загружаем процессор и модель прямо из локальной папки
processor = DetrImageProcessor.from_pretrained(model_path)
model = DetrForObjectDetection.from_pretrained(model_path)

# Переводим в режим оценки (выключает dropout и батч-норм)
model.eval()

# 3. Открываем тестовое изображение
image_path = "00001_aug3.ppm" # Замените на свое фото
image = Image.open(image_path).convert("RGB")

# 4. Подготавливаем картинку
inputs = processor(images=image, return_tensors="pt")

# 5. Делаем предсказание
with torch.no_grad():
    outputs = model(**inputs)

# 6. Конвертируем сырые выходы модели в нормальные координаты COCO [xmin, ymin, xmax, ymax]
# threshold=0.5 означает, что мы берем объекты, в которых модель уверена хотя бы на 50%
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.01)[0]

# 7. Рисуем рамки на картинке
draw = ImageDraw.Draw(image)
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    class_id = label.item()
    class_name = model.config.id2label.get(str(class_id), f"Class {class_id}")
    
    # Рисуем прямоугольник и текст
    draw.rectangle(box, outline="red", width=3)
    draw.text((box[0], box[1] - 10), f"{class_name}: {round(score.item(), 2)}", fill="red")

# Показываем результат
image.show()
# Или сохраняем: image.save("result.jpg")