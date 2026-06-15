# Jetson Nano Deployment Plan

## Cilj

Pokretanje prethodno pripremljenog YOLOv8 sistema na NVIDIA Jetson Nano 2GB ploci.

## Sta se radi na racunaru

- dataset priprema
- trening / fine-tuning
- testiranje modela
- export modela
- priprema koda

## Sta se radi na Jetson Nano

- instalacija minimalnih zavisnosti
- prebacivanje modela i koda
- pokretanje inference-a nad slikom
- pokretanje inference-a nad video snimkom
- merenje FPS-a
- cuvanje screenshotova i demo videa

## Model

Koristi se YOLOv8n zbog ogranicenih resursa Jetson Nano 2GB ploce.

Ulazna rezolucija se smanjuje na 320 ili 416.

ONNX ili TensorRT se koristi kao optimizacija za Jetson Nano.

## Ogranicenja

- nema treninga na Jetson Nano ploci
- nema velikog GUI-ja
- real-time performanse zavise od rezolucije, modela i JetPack okruzenja
