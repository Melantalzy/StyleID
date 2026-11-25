import argparse

def get_args():
    parse = argparse.ArgumentParser(
        description="StyleID style transfer using the Diffusers implementation."
    )
    
    # hyperparameter of StyleID
    parse.add_argument('--T', type=float, default=1.5)
    parse.add_argument('--gamma', type=float, default=0.75)
    parse.add_argument('--without_init_adain', action='store_true')
    parse.add_argument('--without_attn_injection', action='store_true')
    parse.add_argument('--layers', nargs='+', type=int, default=[7, 8, 9, 10, 11])
    
    # hyperparameters of diffusion model
    parse.add_argument('--ddim_steps', type=int, default=20)
    parse.add_argument('--sd_version', type=str, choices=["1.5", "2.0", "2.1-base", "2.1"], default="1.5")
    parse.add_argument('--model_id', type=str, default=None,
                       help='Optional Hugging Face model id overriding --sd_version.')
    parse.add_argument('--device', type=str, choices=["auto", "cuda", "cpu"], default="auto")
    parse.add_argument('--precision', type=str, choices=["float16", "float32"], default="float16")
    parse.add_argument('--resolution', type=int, default=512,
                       help='Center-crop and resize both inputs to this square resolution.')
    parse.add_argument('--local_files_only', action='store_true',
                       help='Load model files from the local Hugging Face cache only.')
    parse.add_argument('--save_intermediates', action='store_true',
                       help='Decode and save every diffusion step (slower and uses more memory).')
    
    # path of content and style iamges
    parse.add_argument('--cnt_fn', type=str, required=True)
    parse.add_argument('--sty_fn', type=str, required=True)
    parse.add_argument('--save_dir', type=str, default='results')
    parse.add_argument('--output_name', type=str, default='stylized_image.jpg',
                       help='Filename for the final stylized image.')
    
    cfg = parse.parse_args()
    return cfg
