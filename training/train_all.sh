#!/bin/bash

PROGRAM="python"

args=(
  "VGG16"
  "VGG19"
  "DenseNet121"
  "MobileNetV3Small"
  "ResNet101V2"
)

for i in 1 2 3 4 5
do
	for arg in "${args[@]}"; do
		echo "Training experiment $arg $i"
		$PROGRAM train.py $arg
	done
done
