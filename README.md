

<div align="center">
<img src="https://raw.githubusercontent.com/Simon-Bertrand/KerPy/main/docs/source/_static/KerPy.png" width="200"/>
</div>



<div align="center"><strong>A Python library to generate 2D-kernels for convolutions, mathematical morphology and more</strong></div>
<br />
<div align="center">
<a href="">Manual installation</a>
<span> · </span>
<a href="https://simon-bertrand.github.io/KerPy/" title="Documentation">Documentation</a>
<span>
</div>

## Features

The features of this project :

- Included famous convolution kernels : Sobel, Prewitt, Finite differences, Laplacian, Gaussian, ...
- Included shaped kernels : circle, triangle, diamond, ...
- Allow easily to pad or stride your generated kernels.
- Contributing : Feel free to ask an implementation of a given kernel or doing it directly.



## Installation

Proceed the installation using pip :
```bash
pip install KernelsPython
```

Then, import the module "kerpy" in Python :
```python
import kerpy
kerpy.diff.finite().divergence()
```
 ```
<kerpy.Kernel numpy =
[[ 0. -1.  0.]
 [-1.  2.  0.]
 [ 0.  0.  0.]] at 0x25fb8477eb0>
 ```

