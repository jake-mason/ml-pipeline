#!/bin/bash

docker build . --tag ml-pipeline:build
docker run --interactive --tty --publish 8888:8888 ml-pipeline:build