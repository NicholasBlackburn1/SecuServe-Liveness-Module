#!/bin/bash 

echo "setting up env..."
source /home/nicky/.cache/pypoetry/virtualenvs/secuserve-liveness-module-kBFFmSxc-py3.8/bin/activate

echo "running liveness"
poetry run python3 SecuServe-Livenes-Module/