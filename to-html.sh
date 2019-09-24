#!/bin/bash

jupyter nbconvert --to html ml-pipeline.ipynb
mv ml-pipeline.html index.html