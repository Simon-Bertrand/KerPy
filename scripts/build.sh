#!/bin/bash
python setup.py bdist_wheel
#sphinx-apidoc -o source ../kerpy &&
cd docs &&  make clean html && make html
