#!/bin/bash
rm -rf build
rm -rf dist
python setup.py version
python setup.py sdist bdist_wheel
twine upload --skip-existing dist/* -u "Simon-Bertrand" -p $1
