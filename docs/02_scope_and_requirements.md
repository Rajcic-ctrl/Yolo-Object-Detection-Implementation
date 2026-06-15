# Scope and Requirements

## Obim projekta

Projekat obuhvata:

- pripremu YOLOv8 modela na racunaru;
- fine-tuning modela za klasu building;
- detekciju ljudi i vozila pomocu pretrained YOLOv8 modela;
- obradu slika;
- obradu video snimaka;
- brojanje objekata;
- merenje FPS-a i inference time-a;
- export modela;
- pokretanje inference dela na Jetson Nano 2GB ploci.

Projekat ne obuhvata:

- razvoj YOLO modela od nule;
- treniranje modela na Jetson Nano ploci;
- deployment na stvarni dron;
- kompleksan korisnicki interfejs;
- profesionalnu produkcionu optimizaciju.

## Funkcionalni zahtevi

FR1: Sistem ucitava sliku.  
FR2: Sistem ucitava video snimak.  
FR3: Sistem ucitava YOLOv8 model.  
FR4: Sistem detektuje ljude i vozila.  
FR5: Sistem detektuje gradjevine kao jednu klasu building.  
FR6: Sistem prikazuje bounding box, klasu i confidence.  
FR7: Sistem broji detektovane objekte.  
FR8: Sistem cuva obradjene rezultate.  
FR9: Sistem meri FPS i inference time.  
FR10: Sistem se moze pokrenuti na Jetson Nano ploci za inference.

## Nefunkcionalni zahtevi

NFR1: Kod mora biti modularan.  
NFR2: Kod mora biti pisan u Pythonu.  
NFR3: Projekat mora imati README fajl.  
NFR4: Projekat mora imati dokumentaciju.  
NFR5: Projekat mora imati demo video.  
