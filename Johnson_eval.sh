#!/bin/bash
model=dance

# Evaluate the final model
mkdir ./images/johnson_output/${model}
for content_img in ./images/content_images/*.jpg
do
  printf "Transfering style of ${model} to ${content_img##*/}\n"
  filename=${content_img##*/}
  python3 Johnson.py eval --content-image ${content_img} --model ./model/${model}.model --output-image ./images/johnson_output/${model}/${filename%%.*}.jpg
done
