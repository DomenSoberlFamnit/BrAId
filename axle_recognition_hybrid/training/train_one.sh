#!/bin/bash

PROGRAM="python"

args=(
  "VGG16"
  "VGG19"
  "DenseNet121"
  "MobileNetV3Small"
  "ResNet101V2"
)

for arg in "${args[@]}"; do
	echo "Training model $arg"
	$PROGRAM train.py $arg
done
