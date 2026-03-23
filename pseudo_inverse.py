from guided_diffusion.unet import create_model
import yaml
import argparse
from utils import *
from operators import *
from guidance import *
from noise_scheduler import NoiseScheduler

def run(args):
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    diffusion_config = config["diffusion_config"]
    train_config = config["train_config"]
    image_index = train_config["image_index"]
    indexes_range = diffusion_config["indexes_range"]

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

    times = list()
    x0 = im2tensor(plt.imread("ffhq256-1k-validation/" + str(image_index).zfill(5) + ".png")).to(device)
    imgshape = x0.shape

    # OPERATORS AND KERNELS
    kernel = torch.tensor(np.loadtxt('kernels/kernel8.txt')).to(device)

    operator_mask = MaskOperator(
        image_shape=imgshape,
        measurement_dim=20,
        mask_type="rectangle",
        device=device
    )

    operator_super_res = SuperResolutionOperator(
        image_shape=imgshape,
        device = device,
    )

    operator_jpeg = JPEG2000Operator(
        image_shape=imgshape,
        quant_step=1.0,
        device=device
        )
    
    operator_motion_blur = MotionBlurOperator(
        kernel=kernel,
        image_shape=imgshape,
        device = device
    )
    

    # Operators
    list_operators = list([operator_motion_blur])
    operator = operator_motion_blur

    sigma_noise = diffusion_config.get("sigma_y", 0.01)
    
    sampler = diffusion_config.get("sampler", "ddim").lower()

    params = {
        "indexes": range(image_index, image_index + indexes_range, 1),
        "model": model,
        "diffusion_config": diffusion_config,
        "config": config,
        "sigma_noise": sigma_noise,
        "device": device,
        "operator": operator,
        "scheduler": scheduler,
    }

    if sampler == "ddpm_pseudo_guidance":

        psnr_list, times = benchmark_denoiser(
            sampler_fn=pseudoinverse_guided_sample_ddpm,
            sampler_name="ddpm_pseudo_guidance",
            **params,
        )

    elif sampler == "ddim_pseudo_guidance":
        
        psnr_list, times = benchmark_denoiser(
            sampler_fn=pseudoinverse_guided_sample_ddim,
            sampler_name="ddim_pseudo_guidance",
            **params,
        )
        

    elif sampler == "ddpm_dps":

        psnr_list, times = benchmark_denoiser(
            sampler_fn=dps_sample_ddpm,
            sampler_name="ddpm_dps",
            **params,
        )

    elif sampler == "ddim_dps":

        psnr_list, times = benchmark_denoiser(
            sampler_fn=dps_sample_ddim,
            sampler_name="ddim_dps",
            **params,
        )

    else :
        
        raise ValueError(
                "Invalid Denoiser, please choose between: "
                "ddpm_pseudo_guidance, ddim_pseudo_guidance, ddpm_dps and ddim_dps"
        )
    
    print(f"Mean ellapsed time is: {np.mean(times):.5}s and std ellapsed time is: {np.std(times):.5}s")
    print(f"Mean psnr is: {np.mean(psnr_list):.5}dB and std psnr is: {np.std(psnr_list):.5}dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", type=str, default="./config.yaml")
    parser.add_argument("--measurement_dim", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="./samples/pigdm_randomH.png")
    parser.add_argument("--pinv_init_path", type=str, default="./samples/pinv_init.png")
    parser.add_argument("--psnr_path", type=str, default="./samples/psnr")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run(args)