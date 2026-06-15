# Dataset Plan

## Klase

- person
- car
- bus
- truck
- motorcycle
- bicycle
- building

## Strategija

Za ljude i vozila koristi se pretrained YOLOv8n model.

Za gradjevine se koristi poseban realan aerial/drone dataset. Sve vrste gradjevina se mapiraju u jednu klasu:

building

## YOLO format anotacija

class_id x_center y_center width height

Za building dataset:

0 x_center y_center width height

## Napomena

Ne koristi se sinteticki generisan dataset.
