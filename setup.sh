#!/bin/bash

# First, obtain the dataset from the UCI machine learning repository
base_url=https://archive.ics.uci.edu/ml/machine-learning-databases/car
for asset in car.names car.data;
	do
		full_url=$base_url/$asset
		wget -q -O data/$asset $full_url
	done

# Next, create a virtualenv
virtualenv venv
source venv/bin/activate

# Next, install the required dependencies
pip3 install -r requirements.txt --no-cache-dir

# Finally, make available to our Jupyter environment the virtualenv we just created
# and start the notebook
ipython kernel install --user --name=venv
jupyter notebook