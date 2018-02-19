#!/bin/bash
git pull

for style_img in ./images/style_images/*.jpg; do
    printf "Training model for ${style_img##*/}\n"
    filename=${style_img##*/}
    python3 Johnson.py train --dataset ./train_data --style-image ${style_img} --save-model-dir ./model --checkpoint-dir ./checkpoint/${filename%%.*}
done

git add *
git commit -m 'Trained Johnson'
git push