#!/bin/bash
python setup.py bdist_wheel

cd docs && sphinx-apidoc -o source ../kerpy &&  make clean html && make html