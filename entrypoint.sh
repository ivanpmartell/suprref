#!/bin/bash
cd ~/suprref/
if [[ -d "venv" ]]
then
echo "Environment already set"
else
virtualenv venv
venv/bin/python setup.py bdist_wheel
venv/bin/pip install dist/suprref-0.1-py3-none-any.whl
fi
source activate
suprref $@