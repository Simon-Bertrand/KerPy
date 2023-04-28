from setuptools import find_packages, setup
setup(
    name='KerPy',
    packages=find_packages(include=['kerpy']),
    version='0.1.0',
    description='A Python library to generate 2D-kernels for convolutions, mathematical morphology and more',
    author='Simon Bertrand',
    license='GPL-3.0',
    install_requires=["numpy"],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)