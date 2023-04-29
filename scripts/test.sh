#!/bin/bash
pylint --recursive=y kerpy
python setup.py pytest
