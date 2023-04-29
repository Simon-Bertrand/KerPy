#!/bin/bash
pylint --recursive=y --fail-under=9 kerpy
python setup.py pytest
