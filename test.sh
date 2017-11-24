#!/bin/bash

for style_img in ./images/style_images/*.jpg; do
  echo "Transfering ${style_img##*/} to cat.jpg"
  python3 Gatys.py cat.jpg ${style_img##*/}
done
