#!/bin/bash 

echo -en "\033]0;Liveness Module\a"
echo "setting up env..."
source /home/nicky/.cache/pypoetry/virtualenvs/secuserve-liveness-module-kBFFmSxc-py3.8/bin/activate

echo "running liveness"
poetry run python3 SecuServe-Livenes-Module/__main__.py