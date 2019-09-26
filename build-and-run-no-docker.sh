#!/bin/bash

# For MacOS
# Installs Homebrew, Python, and related packages

# Grab Homebrew
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
echo 'export PATH="/usr/local/opt/python/libexec/bin:$PATH"' >> ~/.profile
brew install python

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py

pip3 install virtualenv

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