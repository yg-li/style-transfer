#!/bin/bash
model=dance

# Evaluate intermediate checkpoints
# for i in 0 1
# do
#   for ((j=2000; j<=20000; j+=2000))
#   do
#     mkdir ./images/johnson_output/${model}/${i}_$j
#     for content_img in ./images/content_images/*.jpg
#     do
#       printf "Transfering style of ${model} for ${content_img##*/}\n"
#       filename=${content_img##*/}
#       python3 Johnson.py eval --content-image ${content_img} --model ./checkpoint/${model}/${i}_$j.pth --output-image ./images/johnson_output/${model}/${i}_$j/${filename%%.*}.jpg
#     done
#   done
# done

# Evaluate the final model
for content_img in ./images/content_images/*.jpg
do
  printf "Transfering style of ${model} to ${content_img##*/}\n"
  filename=${content_img##*/}
  python3 Johnson.py eval --content-image ${content_img} --model ./model/${model}.model --output-image ./images/johnson_output/${model}/${filename%%.*}.jpg
done
