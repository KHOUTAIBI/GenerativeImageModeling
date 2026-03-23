[Report](https://fr.overleaf.com/6831962659qxwfccgrbhtr#d66384)
![alt text](https://github.com/KHOUTAIBI/GenerativeImageModeling/blob/main/samples/ddim_pseudo_guidance_benchmark_trajectories.png)

# Generative Image Modeling

A modular PyTorch framework for inverse problems in imaging using structured forward operators and generative modeling techniques.

---

## Overview

This repository provides a unified framework to study inverse problems of the form

$$
y = H(x) + \eta
$$

where:
- $x$ is the unknown clean image  
- $H$ is a degradation operator (linear or nonlinear)  
- $\eta$ is optional noise  

The framework focuses on:
- flexible modeling of forward operators  
- composability of degradations  
- compatibility with Plug-and-Play and diffusion-based reconstruction  

---

## Design

All degradations are implemented as operators with a common interface:

- `H(x)` : forward operator  
- `H_pinv(y)` : pseudo-inverse / reconstruction  
- `observe(x)` : generate measurements  

Operators can be **composed** using:

$$
H = H_n \circ \dots \circ H_1
$$

via the `OperatorChain` abstraction.

---

## Implemented Operators

### Linear Operators

#### Masking (Inpainting)

$$
H(x) = M \odot x
$$

- freeform mask generation  
- rectangular masks  
- closed-form guidance (diagonal structure)

---

#### Motion Blur

$$
H(x) = k * x
$$

- implemented using FFT convolution  
- kernel embedding with circular boundary conditions  

Fourier representation:

$$
\widehat{H(x)} = \hat{k} \cdot \hat{x}
$$

Includes:
- adjoint operator $H^\top$
- pseudo-inverse (stabilized inverse filter)
- Wiener filtering

---

### Nonlinear Operators

#### Super-resolution

$$
H(x) = \text{Downsample}(x)
$$

- bicubic / bilinear interpolation  
- inverse via upsampling  

---

#### JPEG Compression

$$
H(x) = \text{JPEG}(x)
$$

- implemented using PIL  
- handles domain conversion:
$$
[-1,1] \leftrightarrow [0,1]
$$

---

#### JPEG2000 (Wavelet Compression)

$$
H(x) = W^{-1}(Q(W(x)))
$$

- multi-level wavelet decomposition  
- coefficient quantization  
- reconstruction via inverse transform  

---

## Operator Composition

Multiple degradations can be combined:

```python
chain = OperatorChain([
    MotionBlurOperator(...),
    JPEGCompressionOperator(...),
    SuperResolutionOperator(...)
])

y = chain.H(x)
x_init = chain.H_pinv(y)
