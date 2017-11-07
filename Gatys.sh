#!/bin/bash
git pull

#for content_img in ./images/content_images/*.jpg; do
  for style_img in ./images/style_images/*.jpg; do
    echo "Transfering ${style_img##*/} to kilbrun.jpg"
    python3 Gatys_implementation.py kilburn.jpg ${style_img##*/}
  done
#done

git add *
git commit -m 'Trained Gatys'
git push
