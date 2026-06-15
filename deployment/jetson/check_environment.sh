#!/bin/bash

echo "Jetson environment check"
echo "------------------------"

echo "JetPack:"
cat /etc/nv_tegra_release 2>/dev/null

echo ""
echo "Python:"
python3 --version

echo ""
echo "Disk:"
df -h

echo ""
echo "Memory:"
free -h

echo ""
echo "CUDA:"
nvcc --version 2>/dev/null

echo ""
echo "TensorRT:"
dpkg -l | grep nvinfer 2>/dev/null
