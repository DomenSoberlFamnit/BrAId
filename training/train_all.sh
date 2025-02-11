#!/bin/bash

PROGRAM="python"

args=(
  "VGG16"
  "VGG19"
  "DenseNet121"
  "MobileNetV3Small"
  "ResNet101V2"
)

for i in 6 7 8 9 10 
do
	for arg in "${args[@]}"; do
		echo "Training experiment $arg $i"
		$PROGRAM train.py $arg $i
	done
done
