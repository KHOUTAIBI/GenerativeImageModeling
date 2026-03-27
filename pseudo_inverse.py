from guided_diffusion.unet import create_model
import yaml
import argparse
import json
import csv
from pathlib import Path

from utils import *
from operators import *
from guidance import *
from noise_scheduler import NoiseScheduler


def _parse_str_list(s: str):
    return [x.strip().lower() for x in s.split(",") if x.strip()]

def _safe_sigma_tag(sigma: float) -> str:
    # folder-friendly
    return f"{sigma:g}".replace(".", "p")

def run(args):
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    diffusion_config = config["diffusion_config"]
    train_config = config["train_config"]
    image_index = train_config["image_index"]
    indexes_range = diffusion_config["indexes_range"]

    # Which samplers to run
    if args.run_all_samplers:
        samplers_to_run = ["ddpm_pseudo_guidance", "ddim_pseudo_guidance", "ddpm_dps", "ddim_dps"]
    elif args.samplers:
        samplers_to_run = _parse_str_list(args.samplers)
    else:
        samplers_to_run = [(args.sampler or diffusion_config.get("sampler", "ddim")).lower()]

    sigma_y_values = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]

    results_dir = Path(args.results_dir or "./results/pseudo_inverse")
    results_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs = {
        "image_size": 256,
        "num_channels": 128,
        "num_res_blocks": 1,
        "channel_mult": "",
        "learn_sigma": True,
        "class_cond": False,
        "use_checkpoint": False,
        "attention_resolutions": 16,
        "num_heads": 4,
        "num_head_channels": 64,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "use_checkpoint": True,
        "dropout": 0.0,
        "resblock_updown": True,
        "use_fp16": False,
        "use_new_attention_order": False,
        "model_path": "./ffhq_10m.pt",
    }

    model = create_model(**model_kwargs)
    model = model.to(device)
    model.eval()

    scheduler = NoiseScheduler(
        num_timesteps=diffusion_config["num_timesteps"],
        beta_init=diffusion_config["beta_init"],
        beta_end=diffusion_config["beta_end"],
        device=device,
    )

    x0 = im2tensor(plt.imread("ffhq256-1k-validation/" + str(image_index).zfill(5) + ".png")).to(device)
    imgshape = x0.shape

    # OPERATORS AND KERNELS
    kernel = torch.tensor(np.loadtxt("kernels/kernel8.txt")).to(device)

    # Mask operator (use CLI measurement_dim if provided)
    mask_measurement_dim = int(args.measurement_dim) if args.measurement_dim is not None else 20
    operator_mask = MaskOperator(
        image_shape=imgshape,
        measurement_dim=mask_measurement_dim,
        mask_type="rectangle",
        device=device,
    )

    operator_motion_blur = MotionBlurOperator(
        kernel=kernel,
        image_shape=imgshape,
        device=device,
    )

    operators_registry = {
        "mask": operator_mask,
        "motion_blur": operator_motion_blur,
    }

    # Which operators to run (default: both)
    if args.operators:
        operator_keys = _parse_str_list(args.operators)
    else:
        operator_keys = ["mask", "motion_blur"]

    unknown_ops = [k for k in operator_keys if k not in operators_registry]
    if unknown_ops:
        raise ValueError(f"Unknown operators: {unknown_ops}. Choose from {list(operators_registry.keys())}")

    operators_to_run = [(k, operators_registry[k]) for k in operator_keys]

    base_params = {
        "indexes": range(image_index, image_index + indexes_range, 1),
        "model": model,
        "diffusion_config": diffusion_config,
        "config": config,
        "device": device,
        "scheduler": scheduler,
        "max_trajectory_frames": 10,
    }

    sampler_fns = {
        "ddpm_pseudo_guidance": pseudoinverse_guided_sample_ddpm,
        "ddim_pseudo_guidance": pseudoinverse_guided_sample_ddim,
        "ddpm_dps": dps_sample_ddpm,
        "ddim_dps": dps_sample_ddim,
    }

    def _run_sampler(sampler_key: str, run_label: str, local_params: dict):
        if sampler_key not in sampler_fns:
            raise ValueError(
                "Invalid Denoiser, please choose between: "
                "ddpm_pseudo_guidance, ddim_pseudo_guidance, ddpm_dps and ddim_dps"
            )
        return benchmark_denoiser(
            sampler_fn=sampler_fns[sampler_key],
            sampler_name=run_label,
            **local_params,
        )

    all_results = []
    for operator_name, operator in operators_to_run:
        for sampler in samplers_to_run:
            for sigma_noise in sigma_y_values:
                run_dir = results_dir / operator_name / sampler / f"sigma_{_safe_sigma_tag(sigma_noise)}"
                run_dir.mkdir(parents=True, exist_ok=True)

                local_params = dict(base_params)
                local_params["operator"] = operator
                local_params["sigma_noise"] = float(sigma_noise)

                # Per-run output paths (used only if benchmark/samplers support saving)
                local_params["output_path"] = str(run_dir / "sample.png")
                local_params["pinv_init_path"] = str(run_dir / "pinv_init.png")
                local_params["psnr_path"] = str(run_dir / "psnr")

                run_label = f"{operator_name}_{sampler}_sigma{sigma_noise:g}"
                psnr_list, times = _run_sampler(
                    sampler_key=sampler,
                    run_label=run_label,
                    local_params=local_params,
                )

                mean_t, std_t = float(np.mean(times)), float(np.std(times))
                mean_p, std_p = float(np.mean(psnr_list)), float(np.std(psnr_list))

                row = {
                    "operator": operator_name,
                    "sampler": sampler,
                    "sigma_y": float(sigma_noise),
                    "mean_time": mean_t,
                    "std_time": std_t,
                    "mean_psnr": mean_p,
                    "std_psnr": std_p,
                    "times": [float(t) for t in times],
                    "psnr": [float(p) for p in psnr_list],
                    "run_dir": str(run_dir),
                }
                all_results.append(row)

                print(f"[{operator_name} | {sampler} | sigma_y={sigma_noise:g}] mean time: {mean_t:.5}s (std {std_t:.5}s)")
                print(f"[{operator_name} | {sampler} | sigma_y={sigma_noise:g}] mean psnr: {mean_p:.5}dB (std {std_p:.5}dB)")

    # Persist results
    (results_dir / "results.json").write_text(json.dumps(all_results, indent=2))

    with open(results_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["operator", "sampler", "sigma_y", "mean_time", "std_time", "mean_psnr", "std_psnr", "run_dir"],
        )
        w.writeheader()
        for r in all_results:
            w.writerow({k: r[k] for k in w.fieldnames})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", type=str, default="./config.yaml")
    parser.add_argument("--measurement_dim", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="./samples/pigdm_randomH.png")
    parser.add_argument("--pinv_init_path", type=str, default="./samples/pinv_init.png")
    parser.add_argument("--psnr_path", type=str, default="./samples/psnr")

    # Sampler controls
    parser.add_argument("--sampler", type=str, default=None,
                        help="Override config diffusion_config.sampler (e.g., ddpm_pseudo_guidance).")
    parser.add_argument("--samplers", type=str, default=None,
                        help="Comma-separated samplers to run (e.g., 'ddpm_dps,ddim_dps').")
    parser.add_argument("--run_all_samplers", action="store_true",
                        help="Run ddpm/ddim × (pseudo_guidance,dps).")

    # Operator controls
    parser.add_argument("--operators", type=str, default=None,
                        help="Comma-separated operators to run: 'mask,motion_blur' (default: both).")

    # Output aggregation
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory to store results.json/results.csv and per-run artifacts.")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run(args)