#!/bin/bash

[[ -d venv ]] ||
	virtualenv venv &&
	source venv/bin/activate &&
	pip3 install -r requirements.txt --no-cache-dir &&
	deactivate

source venv/bin/activate

# Finally, make available to our Jupyter environment the virtualenv we just created
# and start the notebook
ipython kernel install --user --name=venv
jupyter nbconvert ml-pipeline-slides.ipynb --to slides --post serve
