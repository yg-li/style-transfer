#!/bin/bash
for content_img in ./images/content_images/*.jpg; do
  for style_img in ./images/style_images/*.jpg; do
    echo "Transfering ${style_img##*/} to ${content_img##*/}"
    # python3 Gatys_implementation.py ${content_img##*/} ${style_img##*/}
  done
done

git add *
git commit -m 'Testing Gatys.sh'
git push
