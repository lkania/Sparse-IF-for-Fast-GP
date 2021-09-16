#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

pip install virtualenv
virtualenv ifsgp -p python3.8
source ifsgp/bin/activate

pip install -r requirements.txt 
python src/experiments.py
open 3D_synthetic_demo.png
