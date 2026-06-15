# Project Charter

## Naziv projekta

AI sistem za prepoznavanje ljudi, vozila i gradjevina na dron snimcima koriscenjem YOLOv8 modela.

## Namena projekta

Razvoj Python aplikacije koja koristi YOLOv8 model za detekciju objekata na slikama i video snimcima iz dron perspektive.

Kompletan razvoj, trening, fine-tuning i export modela rade se na racunaru, dok se finalni inference/demo izvodi na Jetson Nano 2GB Developer Kit ploci.

## Ciljevi projekta

- Implementacija YOLOv8 modela za object detection.
- Detekcija ljudi, vozila i gradjevina.
- Obrada slike i video snimka.
- Prikaz bounding box-ova, klase i confidence vrednosti.
- Brojanje detektovanih objekata.
- Merenje FPS-a i inference time-a.
- Export modela za Jetson Nano.
- Pokretanje inference dela na Jetson Nano 2GB ploci.
- Izrada dokumentacije, demo videa i prezentacije.

## Stejkholderi

- Profesor / klijent: Prof. dr Aleksandar Peulic
- Projektni tim: 4 studenta
- Projektni menadzer: upisati ime clana tima

## Ogranicenja

- Trening se ne izvodi na Jetson Nano ploci.
- Jetson Nano se koristi samo za inference i demo.
- Zbog 2GB RAM-a koristi se YOLOv8n i smanjena rezolucija ulaza.
- Rok izrade je kratak.
