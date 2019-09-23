#!/bin/bash

# First, create a virtualenv
virtualenv venv
source venv/bin/activate

# Next, install the required dependencies
pip3 install -r requirements.txt --no-cache-dir

# Finally, make available to our Jupyter environment the virtualenv we just created
# and start the notebook
ipython kernel install --user --name=venv
jupyter notebook