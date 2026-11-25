import torch
import numpy as np
import os

from utils import * # image save utils

from stable_diffusion import load_stable_diffusion, encode_latent, decode_latent, get_text_embedding, get_unet_layers, attention_op  # load SD

import cv2
from tqdm import tqdm

from config import get_args
from single_image_metrics import (
    build_run_record,
    evaluate_single_result,
    format_metrics,
    write_run_record,
)


def load_input_image(path, resolution):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {path}")

    height, width = image.shape[:2]
    crop_size = min(height, width)
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    image = image[top:top + crop_size, left:left + crop_size]
    image = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# class for obtain and override the features
class style_transfer_module():
           
    def __init__(self,
        unet, vae, text_encoder, tokenizer, scheduler, cfg, style_transfer_params = None,
    ):  
        
        style_transfer_params_default = {
            'gamma': 0.75,
            'tau': 1.5,
            'injection_layers': [7, 8, 9, 10, 11]
        }
        if style_transfer_params is not None:
            style_transfer_params_default.update(style_transfer_params)
        self.style_transfer_params = style_transfer_params_default
        
        self.unet = unet # SD unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.guidance_scale = 0.0
        self.capture_mode = None

        self.attn_features = {} # where to save key value (attention block feature)
        self.attn_features_modify = {} # where to save key value to modify (attention block feature)

        self.cur_t = None
        
        # Get residual and attention block in decoder
        # [0 ~ 11], total 12 layers
        resnet, attn = get_unet_layers(unet)
        
        # where to inject key and value
        qkv_injection_layer_num = self.style_transfer_params['injection_layers']
    
        
        for i in qkv_injection_layer_num:
            self.attn_features["layer{}_attn".format(i)] = {}
            attn[i].transformer_blocks[0].attn1.register_forward_hook(self.__get_query_key_value("layer{}_attn".format(i)))
        
        # Modify hook (if you change query key value)
        for i in qkv_injection_layer_num:
            attn[i].transformer_blocks[0].attn1.register_forward_hook(self.__modify_self_attn_qkv("layer{}_attn".format(i)))
        
        # triggers for obtaining or modifying features
        
        self.trigger_get_qkv = False # if set True --> save attn qkv in self.attn_features
        self.trigger_modify_qkv = False # if set True --> save attn qkv by self.attn_features_modify
        
        self.modify_num = None # ignore
        self.modify_num_sa = None # ignore
        
    def get_text_condition(self, text):
        if text is None:
            device = self.text_encoder.device
            uncond_input = self.tokenizer(
                [""], padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0].to(device)
            return {'encoder_hidden_states': uncond_embeddings}
        
        text_embeddings, uncond_embeddings = get_text_embedding(text, self.text_encoder, self.tokenizer)
        text_cond = [text_embeddings, uncond_embeddings]
        denoise_kwargs = {
            'encoder_hidden_states': torch.cat(text_cond)
        }
        return denoise_kwargs
    
    def reverse_process(self, input, denoise_kwargs):
        pred_images = []
        pred_latents = []
        
        decode_kwargs = {'vae': self.vae}
        
        # Reverse diffusion process
        for t in tqdm(self.scheduler.timesteps):
            
            # setting t (for saving time step)
            self.cur_t = t.item()
            
            with torch.no_grad():
                
                # For text condition on stable diffusion
                if 'encoder_hidden_states' in denoise_kwargs.keys():
                    bs = denoise_kwargs['encoder_hidden_states'].shape[0]
                    input = torch.cat([input] * bs)
                
                noisy_residual = self.unet(input, t.to(input.device), **denoise_kwargs).sample
                    
                # For text condition on stable diffusion
                if noisy_residual.shape[0] == 2:
                    # perform guidance
                    noise_pred_text, noise_pred_uncond = noisy_residual.chunk(2)
                    noisy_residual = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    input, _ = input.chunk(2)

                scheduler_output = self.scheduler.step(noisy_residual, t, input)
                prev_noisy_sample = scheduler_output.prev_sample
                pred_original_sample = scheduler_output.pred_original_sample
                
                input = prev_noisy_sample
                
                pred_latents.append(pred_original_sample)
                if self.cfg.save_intermediates:
                    pred_images.append(decode_latent(pred_original_sample, **decode_kwargs))

        if not pred_images:
            pred_images.append(decode_latent(pred_latents[-1], **decode_kwargs))
                
        return pred_images, pred_latents
        
            
    ## Inversion (https://github.com/huggingface/diffusion-models-class/blob/main/unit4/01_ddim_inversion.ipynb)
    def invert_process(self, input, denoise_kwargs):

        pred_images = []
        pred_latents = []
        
        decode_kwargs = {'vae': self.vae}

        # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
        timesteps = list(reversed(self.scheduler.timesteps))
        num_inference_steps = len(self.scheduler.timesteps)
        cur_latent = input.clone()

        with torch.no_grad():
            for i in tqdm(range(0, num_inference_steps)):

                t = timesteps[i]
                
                self.cur_t = t.item()
                
                # For text condition on stable diffusion
                if 'encoder_hidden_states' in denoise_kwargs.keys():
                    bs = denoise_kwargs['encoder_hidden_states'].shape[0]
                    cur_latent = torch.cat([cur_latent] * bs)

                # Predict the noise residual
                noise_pred = self.unet(cur_latent, t.to(cur_latent.device), **denoise_kwargs).sample

                # For text condition on stable diffusion
                if noise_pred.shape[0] == 2:
                    # perform guidance
                    noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    cur_latent, _ = cur_latent.chunk(2)

                current_t = max(0, t.item() - (1000//num_inference_steps)) #t
                next_t = t # min(999, t.item() + (1000//num_inference_steps)) # t+1
                alpha_t = self.scheduler.alphas_cumprod[current_t]
                alpha_t_next = self.scheduler.alphas_cumprod[next_t]

                if self.cfg.sd_version == "2.1":
                    beta_t = 1 - alpha_t
                    pred_original_sample = alpha_t.sqrt() * cur_latent - beta_t.sqrt() * noise_pred
                    pred_epsilon = alpha_t.sqrt() * noise_pred + beta_t.sqrt() * cur_latent
                    pred_sample_direction = (1 - alpha_t_next).sqrt() * pred_epsilon
                    cur_latent = alpha_t_next.sqrt() * pred_original_sample + pred_sample_direction
                else:
                    # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
                    cur_latent = (cur_latent - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred
                
                pred_latents.append(cur_latent)
                if self.cfg.save_intermediates:
                    pred_images.append(decode_latent(cur_latent, **decode_kwargs))

        if not pred_images:
            pred_images.append(decode_latent(pred_latents[-1], **decode_kwargs))
                
        return pred_images, pred_latents
        
    # ============================ hook operations ===============================
    
    # save key value in self.original_kv[name]
    def __get_query_key_value(self, name):
        def hook(model, input, output):
            
            if self.trigger_get_qkv:
                _, query, key, value, _ = attention_op(
                    model, input[0], compute_output=False
                )
                if self.capture_mode == "style":
                    feature = (None, key.detach(), value.detach())
                elif self.capture_mode == "content":
                    feature = (query.detach(), None, None)
                else:
                    feature = (query.detach(), key.detach(), value.detach())
                self.attn_features[name][int(self.cur_t)] = feature
            
        return hook

    
    def __modify_self_attn_qkv(self, name):
        def hook(model, input, output):
        
            if self.trigger_modify_qkv:
                _, q_cs, _, _, _ = attention_op(
                    model, input[0], compute_output=False
                )
                
                q_c, k_s, v_s = self.attn_features_modify[name][int(self.cur_t)]
                
                # style injection
                q_hat_cs = q_c * self.style_transfer_params['gamma'] + q_cs * (1 - self.style_transfer_params['gamma'])
                k_cs, v_cs = k_s, v_s
                
                # Replace query key and value
                _, _, _, _, modified_output = attention_op(model, input[0], key=k_cs, value=v_cs, query=q_hat_cs, temperature=self.style_transfer_params['tau'])
                
                return modified_output
        
        return hook

    def take_attn_features(self):
        features = self.attn_features
        self.attn_features = {name: {} for name in features}
        return features
    
    
if __name__ == "__main__":

    cfg = get_args()

    if cfg.resolution <= 0 or cfg.resolution % 8 != 0:
        raise ValueError("--resolution must be a positive multiple of 8")
    if not 0.0 <= cfg.gamma <= 1.0:
        raise ValueError("--gamma must be between 0 and 1")
    if cfg.T <= 0:
        raise ValueError("--T must be greater than 0")

    # Result save at save_dir
    save_dir = cfg.save_dir
    style_image = load_input_image(cfg.sty_fn, cfg.resolution)
    content_image = load_input_image(cfg.cnt_fn, cfg.resolution)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + '/intermediate', exist_ok=True)
    
    # options
    ddim_steps = cfg.ddim_steps
    device = "cuda" if cfg.device == "auto" and torch.cuda.is_available() else cfg.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dtype = torch.float16 if cfg.precision == "float16" else torch.float32
    if device == "cpu" and dtype == torch.float16:
        print("float16 on CPU is unsupported; using float32 instead.")
        dtype = torch.float32
    cfg.runtime_device = device
    cfg.runtime_precision = str(dtype).replace("torch.", "")

    print(f"Device: {device}; precision: {dtype}; resolution: {cfg.resolution}")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    style_text = None
    content_text = None
    
    style_transfer_params = {
        'gamma': cfg.gamma,
        'tau': cfg.T,
        'injection_layers': cfg.layers,
    }
    
    # Get SD modules
    vae, tokenizer, text_encoder, unet, scheduler = load_stable_diffusion(
        sd_version=cfg.sd_version,
        precision_t=dtype,
        device=device,
        model_id=cfg.model_id,
        local_files_only=cfg.local_files_only,
    )
    scheduler.set_timesteps(ddim_steps)
    

    # Init style transfer module
    unet_wrapper = style_transfer_module(unet, vae, text_encoder, tokenizer, scheduler, cfg, style_transfer_params=style_transfer_params)
    
    
    # Get style image tokens
    denoise_kwargs = unet_wrapper.get_text_condition(style_text)
    # The same empty-text embedding is used for all three passes. The text
    # encoder can leave the GPU after this point.
    text_encoder.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()
    
    unet_wrapper.trigger_get_qkv = True # get attention features (key, value)
    unet_wrapper.trigger_modify_qkv = False
    unet_wrapper.capture_mode = "style"
    
    style_latent = encode_latent(normalize(style_image).to(device=vae.device, dtype=dtype), vae)
    
    # invert process
    print("Invert style image...")
    images, latents = unet_wrapper.invert_process(style_latent, denoise_kwargs=denoise_kwargs) # reverse process save activations such as attn, res
    style_latent = latents[-1]
    
    # save image?
    images = [denormalize(image)[0] for image in images]
    image_last = images[-1]
    if cfg.save_intermediates:
        save_image(np.concatenate(images, axis=1), os.path.join(save_dir, "intermediate/inversion_style.jpg"))
    save_image(image_last, os.path.join(save_dir, "intermediate/latent_style.jpg"))
    
    # ================= IMPORTANT =================
    # save key value from style image
    style_features = unet_wrapper.take_attn_features()
    # =============================================
    # Get content image tokens
    unet_wrapper.trigger_get_qkv = True
    unet_wrapper.trigger_modify_qkv = False
    unet_wrapper.capture_mode = "content"
    
    content_latent = encode_latent(normalize(content_image).to(device=vae.device, dtype=dtype), vae)

    # invert process
    print("Invert content image...")
    images, latents = unet_wrapper.invert_process(content_latent, denoise_kwargs=denoise_kwargs) # reverse process save activations such as attn, res
    content_latent = latents[-1]
    
    # save image?
    images = [denormalize(image)[0] for image in images]
    image_last = images[-1]
    if cfg.save_intermediates:
        save_image(np.concatenate(images, axis=1), os.path.join(save_dir, "intermediate/inversion_content.jpg"))
    save_image(image_last, os.path.join(save_dir, "intermediate/latent_content.jpg"))
    
    # ================= IMPORTANT =================
    # save res feature from content image
    content_features = unet_wrapper.take_attn_features()
    # =============================================
    # ================= IMPORTANT =================
    # Set modify features
    for layer_name in style_features.keys():
        unet_wrapper.attn_features_modify[layer_name] = {}
        for t in scheduler.timesteps:
            t = t.item()
            unet_wrapper.attn_features_modify[layer_name][t] = (content_features[layer_name][t][0], style_features[layer_name][t][1], style_features[layer_name][t][2]) # content as q / style as kv        
    # =============================================
    
    unet_wrapper.trigger_get_qkv = False
    unet_wrapper.trigger_modify_qkv = not cfg.without_attn_injection # modify attn feature (key value)
    unet_wrapper.capture_mode = None
    
    if cfg.without_init_adain:
        latent_cs = content_latent
    else:
        latent_cs = (content_latent - content_latent.mean(dim=(2, 3), keepdim=True)) / (content_latent.std(dim=(2, 3), keepdim=True) + 1e-4) * style_latent.std(dim=(2, 3), keepdim=True) + style_latent.mean(dim=(2, 3), keepdim=True)

    # reverse process
    print("Style transfer...")
    images, latents = unet_wrapper.reverse_process(latent_cs, denoise_kwargs=denoise_kwargs) # reverse process save activations such as attn, res
    
    # save image
    images = [denormalize(image)[0] for image in images]
    image_last = images[-1]
    if cfg.save_intermediates:
        save_image(np.concatenate(images, axis=1), os.path.join(save_dir, "reverse_stylized.jpg"))
    if os.path.basename(cfg.output_name) != cfg.output_name:
        raise ValueError("--output_name must be a filename, not a path")
    output_path = os.path.join(save_dir, cfg.output_name)
    save_image(image_last, output_path)

    print(f"Saved result to {os.path.abspath(output_path)}")

    print("Evaluating result...")
    metrics = evaluate_single_result(
        content_image=content_image,
        style_image=style_image,
        stylized_image=image_last,
        vae=vae,
    )
    record = build_run_record(cfg, output_path, metrics)
    json_path, csv_path = write_run_record(record, save_dir, output_path)
    print(format_metrics(metrics))
    print(f"Saved evaluation JSON to {os.path.abspath(json_path)}")
    print(f"Appended evaluation CSV to {os.path.abspath(csv_path)}")

    if device == "cuda":
        peak_gib = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"Peak CUDA tensor memory: {peak_gib:.2f} GiB")
    
