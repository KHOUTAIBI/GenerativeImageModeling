import os
import torch
import matplotlib.pyplot as plt


def predict_x0_from_eps(x_t, eps_theta, alpha_t):
    sqrt_alpha_t = torch.sqrt(torch.clamp(alpha_t, min=1e-12))
    sqrt_one_minus_alpha_t = torch.sqrt(torch.clamp(1.0 - alpha_t, min=0.0))
    x0_hat = (x_t - sqrt_one_minus_alpha_t * eps_theta) / sqrt_alpha_t
    return x0_hat


def _to_vis_map(x):
    """
    x: (B,C,H,W), returns (H,W) map for first image
    """
    x = x[0].detach().float().cpu()
    if x.ndim == 3:
        x = x.abs().mean(dim=0)
    return x


def save_heatmap(tensor, path, title=None):
    vis = _to_vis_map(tensor)
    plt.figure(figsize=(5, 5))
    plt.imshow(vis, cmap="inferno")
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.axis("off")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def tensor_norm(x):
    return x.detach().reshape(x.shape[0], -1).norm(dim=1).mean().item()


def cosine_similarity_map(a, b, eps=1e-12):
    """
    Cosine similarity between batches of tensors
    """
    a_flat = a.detach().reshape(a.shape[0], -1)
    b_flat = b.detach().reshape(b.shape[0], -1)
    dot = (a_flat * b_flat).sum(dim=1)
    na = a_flat.norm(dim=1)
    nb = b_flat.norm(dim=1)
    cos = dot / (na * nb + eps)
    return cos.mean().item()


def compute_input_saliency(x_t, target):
    """
    target should depend on x_t.
    Returns |d target / d x_t|
    """
    grad = torch.autograd.grad(target, x_t, retain_graph=True, allow_unused=True)[0]
    if grad is None:
        return torch.zeros_like(x_t)
    return grad.abs()


def compute_hatx_saliency(model, x_in, t_batch, alpha_t):
    """
    Recompute a fresh forward pass only for saliency,
    so we do not reuse the sampling graph.
    """
    x_sal = x_in.detach().clone().requires_grad_(True)
    eps_sal = model(x_sal, t_batch)[:, :3, :, :]
    hatx_sal = predict_x0_from_eps(x_sal, eps_sal, alpha_t)
    hatx_sal = torch.clamp(hatx_sal, -1.0, 1.0)
    target = (hatx_sal ** 2).sum()
    grad = torch.autograd.grad(target, x_sal, allow_unused=True)[0]
    if grad is None:
        grad = torch.zeros_like(x_sal)
    return grad.abs()


def plot_scalar_logs(logs, outdir):
    os.makedirs(outdir, exist_ok=True)

    for key, values in logs.items():
        if len(values) == 0:
            continue
        plt.figure()
        plt.plot(values)
        plt.grid(True)
        plt.xlabel("Step")
        plt.ylabel(key)
        plt.title(key)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{key}.png"), dpi=150)
        plt.close()