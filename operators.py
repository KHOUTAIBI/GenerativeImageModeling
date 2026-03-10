import torch
import numpy as np

class LinearOperator:
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
        
        self.image_shape = image_shape
        self.n = int(np.prod(image_shape))
        self.m = measurement_dim
        self.device = device

        _, C, H, W = image_shape

        if mask_type == "rectangle":
            hcrop, wcrop = H // 4, W // 2
            corner_top, corner_left = H // 2, int(0.45 * W)

            mask = torch.ones(image_shape, device=device)
            mask[:, :, corner_top:corner_top + hcrop, corner_left:corner_left + wcrop] = 0
            self.mask = mask.to(device)

        elif mask_type == "freeform":
            self.mask = self._generate_freeform_mask(
                image_shape=image_shape,
                num_strokes=num_strokes,
                max_vertices=max_vertices,
                max_brush_width=max_brush_width,
                min_brush_width=min_brush_width,
                max_length=max_length,
                device=device,
            )

        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")

    def flatten(self, x):
        return x.view(x.shape[0], -1)

    def unflatten(self, x):
        return x.view(x.shape[0], *self.image_shape)

    def H(self, x):
        return x * self.mask

    def H_pinv(self, y):
        return y * self.mask

    @torch.no_grad()
    def observe(self, x0, sigma_y=0.0):
        y = self.H(x0)
        if sigma_y > 0:
            y = y + sigma_y * self.mask * torch.randn_like(x0)
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