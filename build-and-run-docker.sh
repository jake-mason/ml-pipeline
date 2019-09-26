#!/bin/bash

docker build . --tag ml-pipeline:build
docker run --interactive --tty --publish 8000:8000 ml-pipeline:build