#!/bin/bash

jupyter nbconvert --to html $1.ipynb
mv $1.html index.html