#!/bin/bash

if [[ -d venv ]]; then
    echo 'Removing previous virtual environment'
    rm -rf venv
fi

virtualenv venv
source venv/bin/activate

pip3 install -r requirements.txt --no-cache-dir