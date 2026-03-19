from guided_diffusion.unet import create_model
from utils import *
from operators import *
from guidance import *
from noise_scheduler import NoiseScheduler

def run(args):
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    diffusion_config = config["diffusion_config"]
    model_config = config["model_config"]
    train_config = config["train_config"]
    image_index = train_config["image_index"]

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

    idx = image_index
    x0 = im2tensor(plt.imread("ffhq256-1k-validation/" + str(idx).zfill(5) + ".png")).to(device)
    imgshape = x0.shape

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
    

    list_operators = list([operator_jpeg, operator_mask, operator_motion_blur])
    operator = OperatorChain(list_operators)

    sigma_noise = diffusion_config.get("sigma_y", 0.01)
    y = operator.observe(x0, sigma_y=sigma_noise)

    save_grid(y, path="./samples/y_init.png")
    save_grid(operator.H(x0), path="./samples/y_clean.png")
    save_grid(x0, path="./samples/x0.png")

    x_init = torch.randn_like(x0)
    save_grid(x_init, args.pinv_init_path, nrow=train_config["num_grid_rows"])

    sampler = diffusion_config.get("sampler", "ddim").lower()

    if sampler == "ddpm":
        x_rec, psnr_list = pseudoinverse_guided_sample_ddpm(
            model=model,
            diffusion_config=diffusion_config,
            operator=operator,
            x0 = x0,
            y=y,
        )

    elif sampler == "ddim":
        x_rec, psnr_list = pseudoinverse_guided_sample_ddim(
            model=model,
            scheduler=scheduler,
            diffusion_config=diffusion_config,
            operator=operator,
            x0 = x0,
            y=y,
        )

    else :
        x_rec, psnr_list = dps_sample_diffsion(
            model=model,
            diffusion_config=diffusion_config,
            operator=operator,
            x0 = x0,
            y=y,
        )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_grid(x_rec, args.output_path, nrow=train_config["num_grid_rows"])
    plt.plot(np.arange(len(psnr_list)), psnr_list)
    plt.grid()
    plt.xlabel("Step")
    plt.ylabel("PSNR")
    plt.savefig(args.psnr_path + f"_{sampler}.png")
    print(f"Saved reconstruction grid to: {args.output_path}")


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