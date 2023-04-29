#!/bin/bash
rm -rf build
rm -rf dist
git fetch -–all -–tags
python setup.py version
python setup.py sdist bdist_wheel
twine upload --skip-existing -r testpypi dist/* -u "Simon-Bertrand" -p $1