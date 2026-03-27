import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import pywt

# -----------------------
# Masking 
# -----------------------

class MaskOperator:
    def __init__(self, 
                image_shape, 
                measurement_dim, 
                mask_type = "freeform",
                num_strokes=15,
                max_vertices=15,
                max_brush_width=14,
                min_brush_width=8,
                max_length=40,
                device="cuda"):
        
        self.type = "linear"  
        self.image_shape = image_shape
        self.n = int(np.prod(image_shape))
        self.m = measurement_dim
        self.device = device

        _, C, H, W = image_shape

        if mask_type == "rectangle":

            hcrop, wcrop = H // 4, W // 4   

            corner_top = int(0.6 * H)       
            corner_left = int(0.45 * W)     

            mask = torch.ones(image_shape, device=device)
            mask[:, :, corner_top:corner_top + hcrop, corner_left:corner_left + wcrop] = 0

            name = "rectangle"

        elif mask_type == "freeform":
            mask = self._generate_freeform_mask(
                image_shape=image_shape,
                num_strokes=num_strokes,
                max_vertices=max_vertices,
                max_brush_width=max_brush_width,
                min_brush_width=min_brush_width,
                max_length=max_length,
                device=device,
            )
            name = "brush_broom"

        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")
    
        self._H = mask
        self.name = name

    def flatten(self, x):
        return x.view(x.shape[0], -1)

    def unflatten(self, x):
        return x.view(x.shape[0], *self.image_shape)

    def H(self, x):
        return x * self._H

    def H_pinv(self, y):
        return y * self._H
    
    def guidance(self, x_t, hatx_t, y, operator , sigma_y, r_t):
        """
        Guidance, moved here to make the code cleaner !
        """
        residual = y - operator.H(hatx_t)
        lam = (sigma_y / r_t).pow(2)
        v = residual / (operator._H + lam)
        u = operator.H(v)   
        inner = (u.detach() * hatx_t).sum()
        guidance = torch.autograd.grad(inner, x_t, retain_graph=True)[0]
        return guidance

    @torch.no_grad()
    def observe(self, x0, sigma_y=0.0):
        y = self.H(x0)
        if sigma_y > 0:
            y = y + sigma_y * self._H * torch.randn_like(x0)
        return y
    
    def _generate_freeform_mask(
        self,
        image_shape,
        num_strokes,
        max_vertices,
        max_brush_width,
        min_brush_width,
        max_length,
        device,
    ):
        _, C, H, W = image_shape

        single_mask = torch.ones((H, W), device=device)

        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )

        for _ in range(num_strokes):
            num_vertices = np.random.randint(1, max_vertices + 1)

            y = np.random.randint(0, H)
            x = np.random.randint(0, W)

            for _ in range(num_vertices):
                angle = np.random.uniform(0, 2 * np.pi)
                length = np.random.randint(10, max_length + 1)
                brush_width = np.random.randint(min_brush_width, max_brush_width + 1)

                y2 = int(np.clip(y + length * np.sin(angle), 0, H - 1))
                x2 = int(np.clip(x + length * np.cos(angle), 0, W - 1))

                self._draw_thick_line(single_mask, yy, xx, y, x, y2, x2, brush_width)

                y, x = y2, x2

        mask = single_mask.unsqueeze(0).unsqueeze(0).repeat(1, C, 1, 1)
        return mask
    
    def _draw_thick_line(self, mask, yy, xx, y0, x0, y1, x1, width):
        dy = y1 - y0
        dx = x1 - x0

        denom = dx * dx + dy * dy
        radius = width / 2.0

        if denom == 0:
            dist2 = (yy - y0) ** 2 + (xx - x0) ** 2
            mask[dist2 <= radius ** 2] = 0.0
            return

        t = ((xx - x0) * dx + (yy - y0) * dy).float() / denom
        t = torch.clamp(t, 0.0, 1.0)

        proj_x = x0 + t * dx
        proj_y = y0 + t * dy

        dist2 = (xx - proj_x) ** 2 + (yy - proj_y) ** 2
        mask[dist2 <= radius ** 2] = 0.0


# ---------------------------
# Superresolution
# ---------------------------
class SuperResolutionOperator:
    def __init__(
        self,
        image_shape,
        measurement_dim=None,
        scale_factor=4,
        device="cpu",
        mode_down="bicubic",
        mode_up="bicubic",
    ):
        
        self.type = "linear"
        self.image_shape = image_shape
        self.n = int(np.prod(image_shape))
        self.m = measurement_dim
        self.device = device
        self.scale_factor = scale_factor
        self.mode_down = mode_down
        self.mode_up = mode_up
        self.name = "super_resolution"

        _, C, H, W = image_shape
        assert H % scale_factor == 0 and W % scale_factor == 0

        self.C = C
        self._H = H
        self.W = W
        self.h = H // scale_factor
        self.w = W // scale_factor

    def flatten(self, x):
        return x.view(x.shape[0], -1)

    def unflatten(self, x):
        return x.view(x.shape[0], *self.image_shape)

    def H(self, x):
        if self.mode_down in ["bilinear", "bicubic"]:
            return F.interpolate(
                x,
                size=(self.h, self.w),
                mode=self.mode_down,
                align_corners=False,
                antialias=True,
            )
        else:
            return F.interpolate(
                x,
                size=(self.h, self.w),
                mode=self.mode_down,
            )

    def H_pinv(self, y):
        if self.mode_up in ["bilinear", "bicubic"]:
            return F.interpolate(
                y,
                size=(self._H, self.W),
                mode=self.mode_up,
                align_corners=False,
            )
        else:
            return F.interpolate(
                y,
                size=(self._H, self.W),
                mode=self.mode_up,
            )

    @torch.no_grad()
    def observe(self, x0, sigma_y=0.0):
        y = self.H(x0)
        if sigma_y > 0:
            y = y + sigma_y * torch.randn_like(y)
        return y
    
# ---------------------------
# JPEG Compression
# ---------------------------
class JPEGCompressionOperator:
    def __init__(self, image_shape, quality=20, device="cpu"):
        self.type = "nonlinear"
        self.image_shape = image_shape
        self.n = int(np.prod(image_shape))
        self.m = self.n
        self.device = device
        self.quality = quality
        self.name = "jpeg_compression"

    def flatten(self, x):
        return x.view(x.shape[0], -1)

    def unflatten(self, x):
        return x.view(x.shape[0], *self.image_shape[1:])

    def _jpeg_single(self, x):
        """
        x: (C,H,W), float in [0,1]
        returns: (C,H,W), float in [0,1]
        """

        x = x.detach().cpu()
        x = ((x + 1.0) / 2.0).clamp(0, 1)
        x_np = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)

        if x_np.shape[2] == 1:
            pil_img = Image.fromarray(x_np[..., 0], mode="L")
        else:
            pil_img = Image.fromarray(x_np, mode="RGB")

        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)

        rec = Image.open(buffer)
        rec_np = np.array(rec)

        if rec_np.ndim == 2:
            rec_np = rec_np[..., None]

        rec_t = torch.from_numpy(rec_np).float() / 255.0
        rec_t = rec_t.permute(2, 0, 1)
        rec_t = 2.0 * rec_t - 1.0
        return rec_t

    def H(self, x):
        outs = [self._jpeg_single(xi) for xi in x]
        return torch.stack(outs, dim=0).to(x.device)

    def H_pinv(self, y):
        return y

    @torch.no_grad()
    def observe(self, x0, sigma_y=0.0):
        return self.H(x0)

# -------------------------
# JPEG2000 Wavelet compression
# -------------------------
class JPEG2000Operator:
    def __init__(
        self,
        image_shape,
        wavelet="haar",
        level=3,
        quant_step=0.02,
        device="cpu",
    ):
        self.type = "nonlinear"
        self.image_shape = image_shape
        self.n = int(np.prod(image_shape))
        self.m = self.n
        self.device = device

        self.wavelet = wavelet
        self.level = level
        self.quant_step = quant_step
        self.name = "jpeg_wavelet"

    def flatten(self, x):
        return x.view(x.shape[0], -1)

    def unflatten(self, x):
        return x.view(x.shape[0], *self.image_shape[1:])

    def _quantize(self, arr, step):
        return np.round(arr / step) * step

    def _jp2_single_channel(self, x2d):
        """
        x2d: (H,W) numpy array in [0,1]
        """
        coeffs = pywt.wavedec2(x2d, wavelet=self.wavelet, level=self.level)

        coeffs_q = [self._quantize(coeffs[0], self.quant_step)]

        for detail_level in coeffs[1:]:
            cH, cV, cD = detail_level
            coeffs_q.append((
                self._quantize(cH, self.quant_step),
                self._quantize(cV, self.quant_step),
                self._quantize(cD, self.quant_step),
            ))

        rec = pywt.waverec2(coeffs_q, wavelet=self.wavelet)
        rec = rec[:x2d.shape[0], :x2d.shape[1]]
        rec = np.clip(rec, 0.0, 1.0)
        return rec

    def _jp2_single(self, x):
        """
        x: (C,H,W), torch tensor in [0,1]
        """
        x = x.detach().cpu()
        x = ((x + 1.0) / 2.0).clamp(0, 1).numpy()

        C, H, W = x.shape

        rec = np.zeros_like(x, dtype=np.float32)
        for c in range(C):
            rec[c] = self._jp2_single_channel(x[c])

        rec = 2.0 * rec - 1.0
        return torch.from_numpy(rec)

    def H(self, x):
        outs = [self._jp2_single(xi) for xi in x]
        return torch.stack(outs, dim=0).to(x.device)

    def H_pinv(self, y):
        return y
   
    @torch.no_grad()
    def observe(self, x0, sigma_y=0.0):
        return self.H(x0)
    

# ----------------------
# Motion Blur operator as seen in the inverse problem TP
# ----------------------
class MotionBlurOperator:
    def __init__(self, kernel, image_shape, device="cpu"):
        """
        kernel:      torch.Tensor of shape (m, n), e.g. (25,25) AS in the tp
        image_shape: (B, C, H, W)
        """
        self.type = "linear"
        self.image_shape = image_shape
        self.n = int(np.prod(image_shape))
        self.m = self.n
        self.device = device
        self.name = "motion_blur"

        B, C, H, W = image_shape
        self.B = B
        self.C = C
        self.Himg = H
        self.Wimg = W

        kernel = kernel.to(device=device, dtype=torch.float32)
        kernel = kernel / kernel.sum()

        self.kernel_small = kernel
        self.kernel_full = self._embed_kernel(kernel, H, W)
        self.fk = torch.fft.fft2(self.kernel_full)          
        self.fk_conj = torch.conj(self.fk)

    def _embed_kernel(self, kernel, H, W):
        """
        Put the small PSF in the top-left corner of an HxW array,
        then roll so that its center is at (0,0).
        """
        m, n = kernel.shape
        k = torch.zeros((H, W), device=kernel.device, dtype=kernel.dtype)
        k[:m, :n] = kernel
        k = torch.roll(k, shifts=(-m // 2, -n // 2), dims=(0, 1))
        return k

    def flatten(self, x):
        return x.view(x.shape[0], -1)

    def unflatten(self, x):
        return x.view(x.shape[0], *self.image_shape[1:])

    def H(self, x):
        """
        Circular convolution by FFT.
        x: (B,C,H,W)
        """
        Xf = torch.fft.fft2(x, dim=(-2, -1))
        Yf = Xf * self.fk[None, None, :, :]
        y = torch.fft.ifft2(Yf, dim=(-2, -1)).real
        return y

    def Ht(self, x):
        """
        Adjoint operator.
        """
        Xf = torch.fft.fft2(x, dim=(-2, -1))
        Yf = Xf * self.fk_conj[None, None, :, :]
        y = torch.fft.ifft2(Yf, dim=(-2, -1)).real
        return y

    def H_pinv(self, y, eps=1e-3):
        """
        Stabilized inverse filter.
        """
        Yf = torch.fft.fft2(y, dim=(-2, -1))
        Xf = Yf / (self.fk[None, None, :, :] + eps)
        x = torch.fft.ifft2(Xf, dim=(-2, -1)).real
        return x

    def wiener(self, y, lam=1e-2):
        """
        More stable than plain inverse filtering.
        """
        Yf = torch.fft.fft2(y, dim=(-2, -1))
        denom = (self.fk.abs() ** 2 + lam)[None, None, :, :]
        Xf = self.fk_conj[None, None, :, :] * Yf / denom
        x = torch.fft.ifft2(Xf, dim=(-2, -1)).real
        return x
    
     
    def guidance(self, x_t, hatx_t, y, operator , sigma_y, r_t):
        """
        Guidance, moved here to make the code cleaner !
        """
        residual = y - operator.H(hatx_t)   
        lam = (sigma_y / r_t).pow(2)
        Rf = torch.fft.fft2(residual, dim=(-2, -1))
        denom = operator.fk.abs() ** 2 + lam   
        Uf = torch.conj(operator.fk)[None, None, :, :] * Rf / denom
        u = torch.fft.ifft2(Uf, dim=(-2, -1)).real

        inner = (u.detach() * hatx_t).sum()
        guidance = torch.autograd.grad(inner, x_t, retain_graph=True)[0]
        return guidance
    
    @torch.no_grad()
    def observe(self, x0, sigma_y=0.0):
        y = self.H(x0)
        if sigma_y > 0:
            y = y + sigma_y * torch.randn_like(y)
        return y

# -----------------
# Operator Chain
# -----------------  
class OperatorChain:
    def __init__(self, operators: list):
        """
        operators: list of operator instances (Mask, JPEG, SR, etc.)
        """
        self.operators = operators
        self.type = "mixed"  # can contain linear + nonlinear

    def __len__(self):
        return len(self.operators)

    def __getitem__(self, idx):
        return self.operators[idx]

    def H(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward chain: H_n  ...  H_2  H_1
        """
        for op in self.operators:
            x = op.H(x)
        return x

    def H_pinv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pseudo-inverse chain: H_1^dag  ...  H_n^dag
        (reverse order)
        """
        for op in reversed(self.operators):
            x = op.H_pinv(x)
        return x

    @torch.no_grad()
    def observe(self, x0: torch.Tensor, sigma_y: float = 0.0) -> torch.Tensor:
        """
        Apply full forward chain (like measurement)
        """
        y = self.H(x0)
        if sigma_y > 0:
            y = y + sigma_y * torch.randn_like(y)
        return y

    def append(self, operator):
        self.operators.append(operator)

    def summary(self):
        print("OperatorChain:")
        for i, op in enumerate(self.operators):
            print(f"  [{i}] {op.__class__.__name__} (type={op.type})")