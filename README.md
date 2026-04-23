# Generative Image Modeling — Guided Diffusion for Inverse Problems

> A research implementation of **training-free guided diffusion** for image restoration.  
> A pre-trained DDPM/DDIM model acts as an image prior and is steered at inference time to solve inverse problems such as inpainting, super-resolution, motion-blur deconvolution, and wavelet compression artefact removal.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Mathematical Background](#3-mathematical-background)
   - 3.1 [Forward Diffusion Process](#31-forward-diffusion-process)
   - 3.2 [Reverse Process — DDPM](#32-reverse-process--ddpm)
   - 3.3 [Faster Sampling — DDIM](#33-faster-sampling--ddim)
   - 3.4 [Inverse Problem Formulation](#34-inverse-problem-formulation)
   - 3.5 [Diffusion Posterior Sampling (DPS)](#35-diffusion-posterior-sampling-dps)
   - 3.6 [Pseudo-Inverse Guided Diffusion (PiGDM)](#36-pseudo-inverse-guided-diffusion-pigdm)
4. [Code Walkthrough](#4-code-walkthrough)
   - 4.1 [noise_scheduler.py](#41-noise_schedulerpy)
   - 4.2 [operators.py](#42-operatorspy)
   - 4.3 [guidance.py](#43-guidancepy)
   - 4.4 [pseudo_inverse.py](#44-pseudo_inversepy)
   - 4.5 [explainability.py](#45-explainabilitypy)
   - 4.6 [config.yaml](#46-configyaml)
5. [Installation & Usage](#5-installation--usage)
6. [Evaluation](#6-evaluation)
7. [References](#7-references)

---

## 1. Project Overview

Most learning-based image restoration models are trained for a single, fixed degradation. This project takes a different route: it uses a **single unconditional diffusion model** (the OpenAI FFHQ 256 × 256 checkpoint, `ffhq_10m.pt`) and, at inference time only, injects measurement consistency into the reverse diffusion trajectory — no retraining needed.

Two guidance algorithms are implemented and can be combined with two samplers:

| Algorithm | Sampler | Key idea |
|---|---|---|
| **DPS** — Diffusion Posterior Sampling | DDPM | Gradient of the reconstruction error |
| **PiGDM** — Pseudo-Inverse Guided Diffusion | DDPM / DDIM | Moore-Penrose projection in the data space |

Four degradation operators are available out of the box: pixel masking, 4× super-resolution, motion blur, and JPEG2000 wavelet compression. They can also be chained arbitrarily via `OperatorChain`.

---

## 2. Repository Structure

```
GenerativeImageModeling/
│
├── noise_scheduler.py       # Linear β-schedule, forward noising, DDIM reverse step
├── guidance.py              # DPS and PiGDM sampling loops (DDPM + DDIM)
├── operators.py             # Degradation operators + OperatorChain
├── pseudo_inverse.py        # CLI entry-point: loads model, runs benchmark
├── explainability.py        # Saliency maps, heatmaps, cosine similarity, scalar logs
├── utils.py                 # PSNR, image I/O, grid saving, benchmark helpers
├── config.yaml              # All hyperparameters
│
├── Diffusion_models.ipynb   # Interactive notebook (training, sampling, visualisation)
│
├── kernels/                 # Pre-computed motion blur kernels (e.g. kernel8.txt)
├── samples/                 # Output directory for generated/restored images
└── sample.png               # Example output
```

> **Submodule:** `guided_diffusion/` — OpenAI's UNet implementation, included as a git submodule.

---

## 3. Mathematical Background

### 3.1 Forward Diffusion Process

The forward process is a fixed Markov chain that progressively destroys the structure of a clean image $x_0$ by adding Gaussian noise over $T$ steps, controlled by a variance schedule $\{\beta_t\}_{t=1}^T$:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1-\beta_t}\, x_{t-1}, \beta_t \mathbf{I}\right)$$

Thanks to the reparameterisation trick, we can jump directly to any noisy step in closed form. Defining $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$:

$$\boxed{x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \mathbf{I})}$$

The code uses a **linear schedule** between $\beta_\text{init}$ and $\beta_\text{end}$:

```python
# noise_scheduler.py
self.betas     = torch.linspace(beta_init, beta_end, num_timesteps)
self.alphas    = 1.0 - self.betas
self.alpha_bar = torch.cumprod(self.alphas, dim=0)
```

As $t \to T$, $\bar\alpha_t \to 0$ and $x_t$ converges to pure Gaussian noise.

---

### 3.2 Reverse Process — DDPM

A UNet $\varepsilon_\theta(x_t, t)$ is trained to predict the noise $\varepsilon$ added at step $t$. From the predicted noise, a clean-image estimate (**Tweedie's formula**) is obtained at each step:

$$\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\varepsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}$$

The DDPM reverse step is:

$$x_{t-1} = \underbrace{\frac{1}{\sqrt{\alpha_t}}\left[x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t,t)\right]}_{\mu_\theta(x_t,t)} + \sqrt{\beta_t}\,z, \quad z\sim\mathcal{N}(0,\mathbf{I})$$

---

### 3.3 Faster Sampling — DDIM

DDIM (Song et al., 2021) introduces a parameter $\eta \in [0,1]$ that interpolates between deterministic and stochastic sampling, and allows skipping timesteps (e.g. 100 instead of 1000):

$$x_s = \sqrt{\bar\alpha_s} \hat{x_0} + c_1\, z + c_2 \varepsilon_\theta$$

where:

$$c_1 = \eta\sqrt{\frac{(1-\bar\alpha_t/\bar\alpha_s)(1-\bar\alpha_s)}{1-\bar\alpha_t}}, \qquad c_2 = \sqrt{1 - \bar\alpha_s - c_1^2}$$

Setting $\eta = 0$ gives a **fully deterministic** trajectory (the default in `config.yaml`), dramatically reducing inference time with little quality loss.

---

### 3.4 Inverse Problem Formulation

We model the degraded measurement $y$ as:

$$y = \mathcal{H}(x_0) + n, \qquad n \sim \mathcal{N}(0, \sigma_y^2 \mathbf{I})$$

where $\mathcal{H}$ is a (possibly non-linear) degradation operator. The goal is to sample from the posterior:

$$p(x_0 \mid y) \propto \underbrace{p(y \mid x_0)}_{\text{likelihood}} \cdot \underbrace{p(x_0)}_{\text{diffusion prior}}$$

without retraining the model. All operators share the interface:

| Method | Description |
|---|---|
| `H(x)` | Forward degradation $\mathcal{H}(x)$ |
| `H_pinv(y)` | Pseudo-inverse $\mathcal{H}^\dagger(y)$ |
| `observe(x0, sigma_y)` | Generate a noisy measurement |
| `guidance(...)` | Closed-form SVD-based guidance (linear operators only) |

---

### 3.5 Diffusion Posterior Sampling (DPS)

DPS (Chung et al., 2022) approximates the score of the likelihood at each reverse step via the Tweedie estimate $\hat x_0$:

$$\nabla_{x_t} \log p(y \mid x_t) \approx -\nabla_{x_t} \left\|\mathcal{H}(\hat x_0) - y\right\|_2^2$$

The guidance is injected as a **normalised gradient step**:

$$x_{t-1}^{\rm DPS} = x_{t-1}^{\rm DDPM} - \zeta \cdot \frac{\nabla_{x_t}\left\|\mathcal{H}(\hat x_0)-y\right\|_2^2}{\left\|\mathcal{H}(\hat x_0)-y\right\|_2}$$

where $\zeta$ is the guidance scale. Because the gradient flows through the UNet via autograd, DPS works for **any differentiable** $\mathcal{H}$, including non-linear ones.

```python
# guidance.py — DPS core
error  = (operator.H(xhat) - y).pow(2).sum()
grad   = torch.autograd.grad(error, x)[0]
x_next = mu + sqrt_beta * z  -  guidance_scale * grad / torch.sqrt(error)
```

---

### 3.6 Pseudo-Inverse Guided Diffusion (PiGDM)

PiGDM (Song et al., 2023) derives a more principled guidance term for **linear** operators by projecting the residual back into the image space using the Moore-Penrose pseudo-inverse $\mathcal{H}^\dagger$:

$$v_t = \mathcal{H}^\dagger y - \mathcal{H}^\dagger\mathcal{H}\,\hat x_0$$

$$g_t = \nabla_{x_t}\left[\langle \text{sg}(v_t), \hat x_0\rangle\right]$$

where $\text{sg}(\cdot)$ is the stop-gradient operator (`.detach()`). The full guided update is:

$$x_{\rm next} = x_{\rm DDIM/DDPM} + \zeta\,\sqrt{\bar\alpha_t} g_t$$

```python
# guidance.py — PiGDM core
v        = operator.H_pinv(y) - operator.H_pinv(operator.H(hatx_t))
inner    = (v.detach() * hatx_t).sum()
guidance = torch.autograd.grad(inner, x_t)[0]
x_next   = x_ddim + guidance_scale * torch.sqrt(alpha_t) * guidance
```

**Noisy linear operators** ($\sigma_y > 0$): each linear operator implements a closed-form Wiener/SVD-based guidance inside its `.guidance()` method, exploiting the spectral structure of $\mathcal{H}$ directly.

**Non-linear operators** (e.g. JPEG2000): the noisy correction is skipped and PiGDM falls back to the simple pseudo-inverse approximation.

---

## 4. Code Walkthrough

### 4.1 `noise_scheduler.py`

**`NoiseScheduler`** is a `nn.Module` that owns the $\beta$-schedule and exposes two methods:

**`add_noise(x0, ε, t)`** — implements the closed-form forward process:

$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t} \varepsilon$$

**`sample_prev_timestep(x_t, ε_θ, t, t_prev, η)`** — implements the full DDIM reverse step and returns both $x_{t_\text{prev}}$ and the intermediate $\hat x_0$. Setting $\eta = 0$ recovers a deterministic trajectory; $\eta = 1$ recovers the DDPM variance level.

---

### 4.2 `operators.py`

#### `MaskOperator`

Supports `"rectangle"` (a fixed crop region) and `"freeform"` (random brush strokes drawn by sampling angles and radii). The mask $m \in \{0,1\}^{C \times H \times W}$ zeroes out corrupted pixels:

$$\mathcal{H}(x) = m \odot x, \qquad \mathcal{H}^\dagger(y) = y \odot m$$

For noisy measurements the guidance solves a per-pixel Wiener-style problem, where $\lambda = (\sigma_y / r_t)^2$:

$$g = \frac{y - \mathcal{H}(\hat x_0)}{m + \lambda}$$

#### `SuperResolutionOperator`

Applies $4\times$ bicubic downsampling as the forward operator and bicubic upsampling as the pseudo-inverse. The diffusion model is responsible for hallucinating the missing high-frequency detail:

$$\mathcal{H}(x) = \text{Bicubic}_\downarrow(x), \qquad \mathcal{H}^\dagger(y) = \text{Bicubic}_\uparrow(y)$$

#### `MotionBlurOperator`

Uses **circular convolution via FFT**, which is both fast and fully differentiable. The PSF kernel is embedded in the top-left corner of an $H \times W$ zero-padded array and rolled to place its centre at the origin:

$$\mathcal{H}(x) = \mathcal{F}^{-1}\left[\mathcal{F}(x) \cdot \mathcal{F}(k)\right]$$

The **stabilised inverse filter** is the default pseudo-inverse:

$$\mathcal{H}^\dagger(y) = \mathcal{F}^{-1}\left[\frac{\mathcal{F}(y)}{\mathcal{F}(k) + \varepsilon}\right]$$

A **Wiener filter** (`wiener()`) is also provided for noisier conditions:

$$\mathcal{H}^\dagger_\lambda(y) = \mathcal{F}^{-1}\left[\frac{\overline{\mathcal{F}(k)}\cdot\mathcal{F}(y)}{|\mathcal{F}(k)|^2 + \lambda}\right]$$

For noisy PiGDM, the same Wiener denominator is reused in the Fourier domain to compute $g_t$ efficiently without leaving frequency space.

#### `JPEG2000Operator`

A **non-linear** operator that applies Haar wavelet decomposition, uniform scalar quantisation of the sub-band coefficients, and wavelet reconstruction:

$$\mathcal{H}(x) = \text{IDWT}\left[\text{Round}\!\left(\frac{\text{DWT}(x)}{\Delta}\right) \cdot \Delta\right]$$

Since quantisation is non-differentiable, $\mathcal{H}^\dagger = \text{id}$ and guidance is computed via the DPS path only.

#### `OperatorChain`

Composes multiple operators: $\mathcal{H} = \mathcal{H}_n \circ \cdots \circ \mathcal{H}_1$. The pseudo-inverse is applied in reverse order: $\mathcal{H}^\dagger = \mathcal{H}_1^\dagger \circ \cdots \circ \mathcal{H}_n^\dagger$.

---

### 4.3 `guidance.py`

Four sampling functions, all following the same loop pattern:

1. Draw $x_T \sim \mathcal{N}(0, \mathbf{I})$
2. For each reversed (sub-)timestep $t$:
   - Predict noise: $\varepsilon_\theta \leftarrow \text{UNet}(x_t, t)$
   - Estimate clean image via Tweedie: $\hat x_0$
   - Compute guidance gradient $g_t$ (DPS or PiGDM)
   - Perform DDPM/DDIM step and add $\zeta \cdot g_t$
   - Record PSNR against the ground truth

| Function | Sampler | Guidance |
|---|---|---|
| `pseudoinverse_guided_sample_ddpm` | DDPM (1 000 steps) | PiGDM |
| `pseudoinverse_guided_sample_ddim` | DDIM (100 steps) | PiGDM |
| `dps_sample_ddpm` | DDPM | DPS |
| `dps_sample_ddim` | DDIM | DPS |

Shared helpers `predict_x0_from_eps` and `ddim_step_from_x0_eps` encapsulate the core formulae, keeping each sampling loop concise.

---

### 4.4 `pseudo_inverse.py`

The CLI entry point that ties everything together:

1. Loads the FFHQ UNet from the `guided_diffusion` submodule.
2. Instantiates a `NoiseScheduler` and the chosen operator from `config.yaml`.
3. Reads the test image from `ffhq256-1k-validation/`.
4. Dispatches to the selected sampler via `benchmark_denoiser`.
5. Reports mean PSNR (dB) ± std and wall-clock time ± std over the test batch.

```bash
python pseudo_inverse.py --config config.yaml --seed 42
```

---

### 4.5 `explainability.py`

Diagnostic tools for auditing guidance behaviour at each diffusion step:

| Function | What it computes |
|---|---|
| `compute_input_saliency(x_t, target)` | $\|\partial\text{target}/\partial x_t\|$ — which pixels drive the reconstruction |
| `compute_hatx_saliency(model, x_t, t, α_t)` | Saliency w.r.t. $\hat x_0$ via a dedicated fresh forward pass |
| `cosine_similarity_map(a, b)` | Alignment between the guidance vector and the noise prediction |
| `save_heatmap(tensor, path)` | Spatial importance map saved with the inferno colormap |
| `plot_scalar_logs(logs, outdir)` | Per-step curves of PSNR, gradient norms, cosine similarity |

```python
from explainability import compute_input_saliency, save_heatmap

saliency = compute_input_saliency(x_t, target=guidance.sum())
save_heatmap(saliency, path="./samples/saliency_t050.png", title="Saliency at t=50")
```

---

### 4.6 `config.yaml`

```yaml
diffusion_config:
  num_timesteps: 1000         # Total diffusion steps (training schedule length)
  beta_init: 0.0001           # β at t=1
  beta_end: 0.02              # β at t=T
  num_inference_steps: 100    # DDIM subsampled steps (ignored for DDPM)
  eta: 0.0                    # 0.0 = deterministic DDIM  |  1.0 = DDPM noise
  sigma_y: 0.1                # Measurement noise level σ_y
  guidance_scale: 0.1         # ζ — guidance correction strength
  sampler: ddim               # ddpm_pseudo_guidance | ddim_pseudo_guidance | ddpm_dps | ddim_dps
  mode: super_res             # Operator tag used for dispatch logic
  save: true                  # Save intermediate samples to samples/

model_config:
  im_channels: 3
  im_size: 256

train_config:
  image_index: 100            # Starting index in ffhq256-1k-validation/
  num_grid_rows: 1
```

---

## 5. Installation & Usage

### Clone with submodule

```bash
git clone --recurse-submodules https://github.com/KHOUTAIBI/GenerativeImageModeling.git
cd GenerativeImageModeling
```

### Install dependencies

```bash
pip install torch torchvision numpy scipy matplotlib tqdm pyyaml PyWavelets Pillow
```

### Download the pre-trained model

```bash
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt \
     -O ffhq_10m.pt
```

### Prepare the validation dataset

Place the FFHQ 256 × 256 validation images at:

```
ffhq256-1k-validation/
  00000.png
  00001.png
  ...
```

### Run inference

```bash
python pseudo_inverse.py --config config.yaml
```

Override specific settings:

```bash
python pseudo_inverse.py --config config.yaml --seed 42 --output_path ./samples/out.png
```

---

## 6. Evaluation

Reconstruction quality is tracked step-by-step via **PSNR** (Peak Signal-to-Noise Ratio):

$$\text{PSNR}(x_0, \hat x) = 10 \cdot \log_{10} \left(\frac{1}{\text{MSE}(x_0, \hat x)}\right) \quad \text{[dB]}$$

Images are normalised to $[-1, 1]$. Higher PSNR means the reconstruction is closer to the ground truth. The script prints **mean ± std** over the evaluated image set and also saves per-step PSNR curves if `save: true`.

---

## 7. References

| Paper | Authors | Venue | Link |
|---|---|---|---|
| **DDPM** — Denoising Diffusion Probabilistic Models | Ho et al. | NeurIPS 2020 | [arxiv 2006.11239](https://arxiv.org/abs/2006.11239) |
| **DDIM** — Denoising Diffusion Implicit Models | Song et al. | ICLR 2021 | [arxiv 2010.02502](https://arxiv.org/abs/2010.02502) |
| **DPS** — Diffusion Posterior Sampling for General Noisy Inverse Problems | Chung et al. | ICLR 2023 | [arxiv 2209.14687](https://arxiv.org/abs/2209.14687) |
| **PiGDM** — Pseudoinverse-Guided Diffusion Models for Inverse Problems | Song et al. | ICLR 2023 | [arxiv 2305.10483](https://arxiv.org/abs/2305.10483) |
| **Guided Diffusion** — Diffusion Models Beat GANs on Image Synthesis | Dhariwal & Nichol | NeurIPS 2021 | [arxiv 2105.05233](https://arxiv.org/abs/2105.05233) |
