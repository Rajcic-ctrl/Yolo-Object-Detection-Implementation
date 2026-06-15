# Jetson Nano Setup

Ovaj folder sadrzi fajlove za pokretanje inference dela projekta na Jetson Nano 2GB ploci.

## Provera okruzenja

Pokrenuti na Jetson Nano ploci:

cat /etc/nv_tegra_release
python3 --version
df -h
free -h

Ako postoji CUDA:

nvcc --version

Ako postoji TensorRT:

dpkg -l | grep nvinfer

## Napomena

Trening i fine-tuning se rade na racunaru.

Jetson Nano se koristi samo za inference, FPS merenje i demo.
