# Risks and Assumptions

## Pretpostavke

- Projekat se razvija na racunaru.
- Trening i fine-tuning se izvode na racunaru.
- Jetson Nano 2GB se koristi za inference i demo.
- Koristi se YOLOv8n zbog ogranicenih resursa.
- Gradjevine se tretiraju kao jedna klasa.

## Rizici

1. Nedostatak kvalitetnog building dataset-a.
   Resenje: koristiti manji realan aerial dataset i dokumentovati ogranicenja.

2. Slab FPS na Jetson Nano 2GB ploci.
   Resenje: koristiti YOLOv8n, manju rezoluciju i ONNX/TensorRT optimizaciju.

3. Problemi sa instalacijom biblioteka.
   Resenje: koristiti requirements fajlove i jasno uputstvo.

4. Kratak rok.
   Resenje: prvo zavrsiti osnovnu detekciju slike i videa, zatim Jetson demo.

5. Los rezultat za building klasu.
   Resenje: prikazati building model kao poseban deo i jasno opisati ogranicenja.
