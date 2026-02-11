#!/bin/bash 

python -m venv venv && \
source venv/bin/activate && \
python -m pip install _deps/memilio-src/pycode/memilio-epidata
