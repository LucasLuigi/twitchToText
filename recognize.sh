#!/bin/bash

python3 text_recognition.py --east frozen_east_text_detection.pb -i sample.png -w 1024 -e 512 -c 0.05 -p 0.05 -m 7