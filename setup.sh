#!/bin/bash

base_url=https://archive.ics.uci.edu/ml/machine-learning-databases/car

for asset in car.names car.data;
	do
		full_url=$base_url/$asset
		wget -q -O data/$asset $full_url
	done