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
    parse.add_argument(
        '--trajectory_mode',
        choices=[
            'baseline', 'dynamic', 'token_dynamic', 'output_fusion', 'transport'
        ],
        default='baseline',
        help=(
            'baseline keeps the original fixed StyleID query blend; dynamic '
            'uses generation-aware three-trajectory alignment.'
        ),
    )
    parse.add_argument('--trajectory_drift_threshold', type=float, default=0.4)
    parse.add_argument('--trajectory_min_generation_weight', type=float, default=0.1)
    parse.add_argument('--trajectory_max_generation_weight', type=float, default=0.4)
    parse.add_argument('--trajectory_guard_strength', type=float, default=0.0)
    parse.add_argument('--trajectory_min_style_gate', type=float, default=1.0)
    parse.add_argument('--trajectory_start_fraction', type=float, default=0.2)
    parse.add_argument('--trajectory_end_fraction', type=float, default=0.8)
    parse.add_argument('--trajectory_token_temperature', type=float, default=1.0)
    parse.add_argument('--trajectory_temporal_smoothing', type=float, default=0.0)
    parse.add_argument('--trajectory_output_strength', type=float, default=1.0)
    parse.add_argument('--transport_prototype_side', type=int, default=8)
    parse.add_argument('--transport_temperature', type=float, default=0.07)
    parse.add_argument('--transport_temporal_weight', type=float, default=0.25)
    parse.add_argument('--transport_velocity', type=float, default=0.5)
    parse.add_argument('--transport_coverage_strength', type=float, default=0.3)
    parse.add_argument('--transport_sinkhorn_iterations', type=int, default=3)
    parse.add_argument('--transport_max_tokens', type=int, default=1024)
    parse.add_argument('--transport_min_tokens', type=int, default=1)
    parse.add_argument('--transport_confidence_power', type=float, default=0.0)
    parse.add_argument(
        '--save_trajectory_diagnostics',
        action='store_true',
        help='Save per-layer, per-timestep query drift and alignment weights.',
    )
    
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
