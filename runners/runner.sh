#!/bin/bash

if ! [[ -d ../setup/venv ]]; then
    cd ../setup && sh create_venv.sh
fi

source ../setup/venv/bin/activate

