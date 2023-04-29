from setuptools import find_packages, setup
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]
setup(
    name='KerPy',
    packages=find_packages(include=['kerpy']),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='A Python library to generate 2D-kernels for convolutions, mathematical morphology and more',
    author='Simon Bertrand',
    author_email="simonbertrand.contact@gmail.com",
    license='GPL-3.0',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
    test_suite='tests',
)