#!/bin/bash
mkdir -p data/output
./gpu_image_proc data/input/lena.png data/output/lena_gray.png
echo "Output saved in data/output/"
