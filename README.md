# Детекция дорожных знаков: сравнение 5 моделей

Этот проект представляет собой законченное **CV-решение (Computer Vision)** полного цикла, разработанное в рамках учебной практики БВТ. 
**Цель работы:** обучить, протестировать и сравнить эффективность пяти популярных архитектур в задаче локализации и классификации дорожных объектов.

## Сравниваемые модели

| Модель | Тип архитектуры | Backbone | Веса |
|--------|-----------------|----------|------|
| 🟣 **YOLOv8** | anchor-free, одностадийная | CSPDarknet | `yolov8s.pt` |
| 🔴 **YOLOv5u** | anchor-free, одностадийная | CSPDarknet | `yolov5su.pt` |
| 🔵 **Faster R-CNN** | двухстадийная | ResNet-50 + FPN | `fasterrcnn_resnet50_fpn` |
| 🟢 **SSD300** | одностадийная, anchor-based | VGG-16 | `ssd300_vgg16` |
| 🟡 **RetinaNet** | одностадийная, focal loss | ResNet-50 + FPN | `retinanet_resnet50_fpn` |

В рамках исследования проводится fine-tuning (дообучение) пяти современных моделей. В качестве отправной точки используются веса, предобученные на масштабном датасете COCO.

> **Примечание.** Используется **YOLOv5u**, а не классический YOLOv5 — суффикс `u` у Ultralytics означает anchor-free голову (как у YOLOv8), а не оригинальную anchor-based версию.

## Структура репозитория

Архитектура проекта спроектирована по модульному принципу, разделяя логику обработки данных, построения моделей и самого пайплайна обучения:
```text
ML-Practicum/
├── configs/            # YAML-конфиги экспериментов (гиперпараметры, пути)
│   └── default.yaml
├── data/
│   ├── raw/            # GTSDB: изображения + gt.txt (кладётся вручную)
│   └── processed/      # конвертированные аннотации (YOLO / JSON-манифест)
├── src/
│   ├── dataset/        # конвертация в YOLO/манифест, аугментации, Dataset
│   ├── models/         # 5 моделей детекции + общий интерфейс
│   ├── training/       # цикл обучения, логирование, чекпойнты
│   ├── evaluation/     # метрики (mAP, Precision, Recall, F1)
│   └── utils/          # визуализация, инференс на фото
├── notebooks/          # исследовательский анализ данных (EDA)
├── results/            # графики, логи, чекпойнты
└── main.py             # точка входа (prepare / train / eval / predict)
```

## Установка
Пайплайн оптимизирован под обучение на GPU. Перед запуском убедитесь в наличии установленных драйверов CUDA.
```bash
# 1. PyTorch с CUDA под вашу GPU (пример: GTX 1660 Super, CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Остальные зависимости
pip install -r requirements.txt

# 3. Проверка, что GPU доступен
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Использование

```bash
# Подготовка данных: конвертация "сырого" GTSDB (data/raw/) в YOLO + JSON-манифест
python main.py --mode prepare --config configs/default.yaml

# Обучение одной модели
python main.py --mode train --model yolov8 --config configs/default.yaml

# Обучение всех 5 моделей
python main.py --mode train --model all --config configs/default.yaml

# Оценка и сравнение моделей (таблицы + графики в results/)
python main.py --mode eval --model all --config configs/default.yaml

# Инференс на изображение
python main.py --mode predict --model yolov8 --source path/to/image.jpg
```

## Датасет

[GTSDB (German Traffic Sign Detection Benchmark)](https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip) —
изображения с разметкой дорожных знаков, 900 изображений, 43 класса.

Изображения и файл разметки `gt.txt` уже содержатся в проекте в директории `data/raw/`.