# System Architecture

## PC / laptop deo

- dataset priprema
- YOLOv8 trening / fine-tuning
- testiranje modela
- export modela

## Jetson Nano deo

- ucitavanje slike ili video snimka
- YOLO inference
- prikaz bounding box-ova
- brojanje objekata
- FPS / inference time
- demo output

## Tok sistema

PC:
Dataset -> YOLOv8 trening/test -> export modela

Jetson Nano:
Model -> OpenCV input -> YOLO inference -> vizualizacija -> FPS/log/demo output

## Modeli

- YOLOv8n pretrained za ljude i vozila
- YOLOv8n fine-tuned za klasu building

## Klase

- person
- car
- bus
- truck
- motorcycle
- bicycle
- building
