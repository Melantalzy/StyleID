import torch
import torch.nn.functional as F
import numpy as np
import os
import json

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
            'injection_layers': [7, 8, 9, 10, 11],
            'trajectory_mode': 'baseline',
            'trajectory_drift_threshold': 0.4,
            'trajectory_min_generation_weight': 0.1,
            'trajectory_max_generation_weight': 0.4,
            'trajectory_guard_strength': 0.0,
            'trajectory_min_style_gate': 1.0,
            'trajectory_start_fraction': 0.2,
            'trajectory_end_fraction': 0.8,
            'trajectory_token_temperature': 1.0,
            'trajectory_temporal_smoothing': 0.0,
            'trajectory_output_strength': 1.0,
            'transport_prototype_side': 8,
            'transport_temperature': 0.07,
            'transport_temporal_weight': 0.25,
            'transport_velocity': 0.5,
            'transport_coverage_strength': 0.3,
            'transport_sinkhorn_iterations': 3,
            'transport_max_tokens': 1024,
            'transport_min_tokens': 1,
            'transport_confidence_power': 0.0,
            'save_trajectory_diagnostics': False,
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
        self.trajectory_diagnostics = []
        self.trajectory_weight_ema = {}
        self.trajectory_transport_state = {}
        
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

    @staticmethod
    def _pool_style_prototypes(features, prototype_side):
        batch_heads, token_count, head_dim = features.shape
        spatial_side = int(round(token_count ** 0.5))
        target_side = min(prototype_side, spatial_side)
        if spatial_side * spatial_side == token_count:
            spatial = features.transpose(1, 2).reshape(
                batch_heads, head_dim, spatial_side, spatial_side
            )
            pooled = F.adaptive_avg_pool2d(
                spatial, (target_side, target_side)
            )
            return pooled.flatten(2).transpose(1, 2)

        pooled = F.adaptive_avg_pool1d(
            features.transpose(1, 2), target_side * target_side
        )
        return pooled.transpose(1, 2)

    @staticmethod
    def _project_attention_heads(model, hidden_states, attention_heads):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
        else:
            batch_size = hidden_states.shape[0]

        projected = model.batch_to_head_dim(attention_heads)
        projected = model.to_out[0](projected)
        projected = model.to_out[1](projected)
        if input_ndim == 4:
            projected = projected.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        if model.residual_connection:
            projected = projected + residual
        return projected / model.rescale_output_factor

    def _transport_attention_output(self, name, model, hidden_states, q_c, q_g, k_s, v_s):
        prototype_side = self.style_transfer_params['transport_prototype_side']
        k_proto = self._pool_style_prototypes(k_s, prototype_side)
        v_proto = self._pool_style_prototypes(v_s, prototype_side)

        q_c_normalized = F.normalize(q_c.float(), dim=-1, eps=1e-8)
        q_g_normalized = F.normalize(q_g.float(), dim=-1, eps=1e-8)
        k_normalized = F.normalize(k_proto.float(), dim=-1, eps=1e-8)
        inverse_temperature = 1.0 / self.style_transfer_params[
            'transport_temperature'
        ]
        content_logits = torch.bmm(
            q_c_normalized, k_normalized.transpose(1, 2)
        ) * inverse_temperature
        generation_logits = torch.bmm(
            q_g_normalized, k_normalized.transpose(1, 2)
        ) * inverse_temperature
        content_map = torch.softmax(content_logits, dim=-1)
        generation_map = torch.softmax(generation_logits, dim=-1)

        mean_map = 0.5 * (content_map + generation_map)
        correspondence_js = 0.5 * (
            (
                content_map
                * (
                    content_map.clamp_min(1e-8).log()
                    - mean_map.clamp_min(1e-8).log()
                )
            ).sum(dim=-1)
            + (
                generation_map
                * (
                    generation_map.clamp_min(1e-8).log()
                    - mean_map.clamp_min(1e-8).log()
                )
            ).sum(dim=-1)
        )
        correspondence_confidence = (
            1.0 - correspondence_js / np.log(2.0)
        ).clamp(0.0, 1.0)

        eps = 1e-8
        content_log = content_map.clamp_min(eps).log()
        generation_log = generation_map.clamp_min(eps).log()
        previous_state = self.trajectory_transport_state.get(name)
        if previous_state is None:
            prediction_log = content_log
        else:
            previous_map, previous_content_map = previous_state
            previous_map = previous_map.float()
            previous_content_map = previous_content_map.float()
            prediction_log = previous_map.clamp_min(eps).log() + (
                self.style_transfer_params['transport_velocity']
                * (
                    content_log
                    - previous_content_map.clamp_min(eps).log()
                )
            )

        temporal_weight = self.style_transfer_params[
            'transport_temporal_weight'
        ]
        current_weight = (1.0 - temporal_weight) * 0.5
        transport_logits = (
            current_weight * content_log
            + current_weight * generation_log
            + temporal_weight * prediction_log
        )
        row_map = torch.softmax(transport_logits, dim=-1)

        balanced_map = row_map
        query_count = balanced_map.shape[-2]
        prototype_count = balanced_map.shape[-1]
        target_column_mass = query_count / max(prototype_count, 1)
        for _ in range(
            self.style_transfer_params['transport_sinkhorn_iterations']
        ):
            balanced_map = balanced_map / balanced_map.sum(
                dim=-1, keepdim=True
            ).clamp_min(eps)
            balanced_map = balanced_map / balanced_map.sum(
                dim=-2, keepdim=True
            ).clamp_min(eps)
            balanced_map = balanced_map * target_column_mass
        balanced_map = balanced_map / balanced_map.sum(
            dim=-1, keepdim=True
        ).clamp_min(eps)

        coverage_strength = self.style_transfer_params[
            'transport_coverage_strength'
        ]
        transport_map = (
            (1.0 - coverage_strength) * row_map
            + coverage_strength * balanced_map
        )
        transport_map = transport_map / transport_map.sum(
            dim=-1, keepdim=True
        ).clamp_min(eps)
        self.trajectory_transport_state[name] = (
            transport_map.detach().to(dtype=torch.float16),
            content_map.detach().to(dtype=torch.float16),
        )

        transported_heads = torch.bmm(
            transport_map, v_proto.float()
        ).to(dtype=q_g.dtype)
        transported_output = self._project_attention_heads(
            model, hidden_states, transported_heads
        )
        normalized_entropy = (
            -(
                transport_map
                * transport_map.clamp_min(eps).log()
            ).sum(dim=-1).mean()
            / np.log(max(prototype_count, 2))
        )
        return transported_output, normalized_entropy, correspondence_confidence
        
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

                mode = self.style_transfer_params['trajectory_mode']
                gamma = self.style_transfer_params['gamma']
                generation_weight = q_cs.new_tensor(1.0 - gamma)
                style_gate = q_cs.new_tensor(1.0)
                drift = None
                total_train_steps = self.scheduler.config.num_train_timesteps
                denoise_progress = 1.0 - int(self.cur_t) / max(
                    total_train_steps - 1, 1
                )
                dynamic_active = (
                    mode in (
                        'dynamic', 'token_dynamic', 'output_fusion', 'transport'
                    )
                    and self.style_transfer_params['trajectory_start_fraction']
                    <= denoise_progress
                    <= self.style_transfer_params['trajectory_end_fraction']
                )

                should_measure = (
                    mode in (
                        'dynamic', 'token_dynamic', 'output_fusion', 'transport'
                    )
                    or self.style_transfer_params['save_trajectory_diagnostics']
                )
                if should_measure:
                    drift = (
                        1.0
                        - F.cosine_similarity(
                            q_cs.float(), q_c.float(), dim=-1, eps=1e-8
                        )
                    ).mean().clamp(0.0, 2.0)

                weight_min = generation_weight
                weight_max = generation_weight
                weight_std = generation_weight.new_zeros(())
                fraction_below_baseline = generation_weight.new_zeros(())
                fraction_above_baseline = generation_weight.new_zeros(())

                if dynamic_active and mode == 'dynamic':
                    threshold = self.style_transfer_params[
                        'trajectory_drift_threshold'
                    ]
                    max_generation_weight = self.style_transfer_params[
                        'trajectory_max_generation_weight'
                    ]
                    min_generation_weight = self.style_transfer_params[
                        'trajectory_min_generation_weight'
                    ]
                    content_confidence = 1.0 - (drift / threshold).clamp(
                        0.0, 1.0
                    )
                    generation_weight = (
                        min_generation_weight
                        + (max_generation_weight - min_generation_weight)
                        * content_confidence
                    )
                    style_gate = torch.exp(
                        -self.style_transfer_params['trajectory_guard_strength']
                        * drift
                    ).clamp(
                        min=self.style_transfer_params['trajectory_min_style_gate'],
                        max=1.0,
                    )
                    weight_min = generation_weight
                    weight_max = generation_weight
                    fraction_below_baseline = (
                        generation_weight < (1.0 - gamma)
                    ).float()
                    fraction_above_baseline = (
                        generation_weight > (1.0 - gamma)
                    ).float()

                if dynamic_active and mode in ('token_dynamic', 'output_fusion'):
                    token_drift = 1.0 - F.cosine_similarity(
                        q_cs.float(), q_c.float(), dim=-1, eps=1e-8
                    )
                    token_drift = token_drift.clamp(0.0, 2.0)
                    quartiles = torch.quantile(
                        token_drift,
                        token_drift.new_tensor([0.25, 0.5, 0.75]),
                        dim=-1,
                        keepdim=True,
                    )
                    q25, q50, q75 = quartiles.unbind(dim=0)
                    robust_scale = (q75 - q25).clamp_min(0.02)
                    min_generation_weight = self.style_transfer_params[
                        'trajectory_min_generation_weight'
                    ]
                    max_generation_weight = self.style_transfer_params[
                        'trajectory_max_generation_weight'
                    ]
                    weight_range = max_generation_weight - min_generation_weight
                    if weight_range > 0:
                        baseline_ratio = (
                            (1.0 - gamma) - min_generation_weight
                        ) / weight_range
                        baseline_ratio = min(max(baseline_ratio, 1e-4), 1.0 - 1e-4)
                        baseline_logit = np.log(
                            baseline_ratio / (1.0 - baseline_ratio)
                        )
                        normalized_drift = (q50 - token_drift) / (
                            robust_scale
                            * self.style_transfer_params[
                                'trajectory_token_temperature'
                            ]
                        )
                        token_confidence = torch.sigmoid(
                            normalized_drift + baseline_logit
                        )
                        generation_weight = (
                            min_generation_weight
                            + weight_range * token_confidence
                        )

                        smoothing = self.style_transfer_params[
                            'trajectory_temporal_smoothing'
                        ]
                        previous_weight = self.trajectory_weight_ema.get(name)
                        if (
                            smoothing > 0.0
                            and previous_weight is not None
                            and previous_weight.shape == generation_weight.shape
                        ):
                            generation_weight = (
                                smoothing * previous_weight
                                + (1.0 - smoothing) * generation_weight
                            )
                        self.trajectory_weight_ema[name] = generation_weight.detach()

                    base_weight = 1.0 - gamma
                    weight_min = generation_weight.min()
                    weight_max = generation_weight.max()
                    weight_std = generation_weight.float().std(unbiased=False)
                    fraction_below_baseline = (
                        generation_weight < base_weight
                    ).float().mean()
                    fraction_above_baseline = (
                        generation_weight > base_weight
                    ).float().mean()

                if generation_weight.ndim > 0:
                    token_weight = generation_weight[..., None].to(
                        device=q_c.device, dtype=q_c.dtype
                    )
                    q_hat_cs = q_c * (1.0 - token_weight) + q_cs * token_weight
                else:
                    q_hat_cs = (
                        q_c * (1.0 - generation_weight)
                        + q_cs * generation_weight
                    )

                _, _, _, _, baseline_output = attention_op(
                    model,
                    input[0],
                    key=k_s,
                    value=v_s,
                    query=q_hat_cs,
                    temperature=self.style_transfer_params['tau'],
                )

                modified_output = baseline_output
                output_change_ratio = baseline_output.new_zeros(())
                transport_entropy = baseline_output.new_zeros(())
                transport_confidence = baseline_output.new_zeros(())
                if dynamic_active and mode == 'output_fusion':
                    _, _, _, _, content_output = attention_op(
                        model,
                        input[0],
                        key=k_s,
                        value=v_s,
                        query=q_c,
                        temperature=self.style_transfer_params['tau'],
                    )
                    _, _, _, _, generation_output = attention_op(
                        model,
                        input[0],
                        key=k_s,
                        value=v_s,
                        query=q_cs,
                        temperature=self.style_transfer_params['tau'],
                    )

                    if generation_weight.ndim > 0:
                        batch_size = q_cs.shape[0] // model.heads
                        output_weight = generation_weight.view(
                            batch_size, model.heads, generation_weight.shape[-1]
                        ).mean(dim=1)
                        output_weight = output_weight.to(
                            device=content_output.device,
                            dtype=content_output.dtype,
                        )
                        if content_output.ndim == 3:
                            output_weight = output_weight[..., None]
                        else:
                            height, width = content_output.shape[-2:]
                            output_weight = output_weight.view(
                                batch_size, 1, height, width
                            )
                    else:
                        output_weight = generation_weight.to(
                            device=content_output.device,
                            dtype=content_output.dtype,
                        )

                    fused_output = (
                        content_output * (1.0 - output_weight)
                        + generation_output * output_weight
                    )
                    output_strength = self.style_transfer_params[
                        'trajectory_output_strength'
                    ]
                    modified_output = baseline_output + output_strength * (
                        fused_output - baseline_output
                    )
                    output_change_ratio = (
                        (modified_output.float() - baseline_output.float()).norm()
                        / baseline_output.float().norm().clamp_min(1e-8)
                    )

                transport_active = (
                    dynamic_active
                    and mode == 'transport'
                    and self.style_transfer_params['transport_min_tokens']
                    <= q_cs.shape[1]
                    <= self.style_transfer_params['transport_max_tokens']
                )
                if transport_active:
                    (
                        transported_output,
                        transport_entropy,
                        correspondence_confidence,
                    ) = (
                        self._transport_attention_output(
                            name,
                            model,
                            input[0],
                            q_c,
                            q_cs,
                            k_s,
                            v_s,
                        )
                    )
                    output_strength = self.style_transfer_params[
                        'trajectory_output_strength'
                    ]
                    confidence_power = self.style_transfer_params[
                        'transport_confidence_power'
                    ]
                    batch_size = q_cs.shape[0] // model.heads
                    confidence_gate = correspondence_confidence.view(
                        batch_size,
                        model.heads,
                        correspondence_confidence.shape[-1],
                    ).mean(dim=1)
                    confidence_gate = confidence_gate.pow(confidence_power).to(
                        device=baseline_output.device,
                        dtype=baseline_output.dtype,
                    )
                    if baseline_output.ndim == 3:
                        confidence_gate = confidence_gate[..., None]
                    else:
                        height, width = baseline_output.shape[-2:]
                        confidence_gate = confidence_gate.view(
                            batch_size, 1, height, width
                        )
                    modified_output = baseline_output + (
                        output_strength
                        * confidence_gate
                        * (transported_output - baseline_output)
                    )
                    transport_confidence = correspondence_confidence.mean()
                    output_change_ratio = (
                        (modified_output.float() - baseline_output.float()).norm()
                        / baseline_output.float().norm().clamp_min(1e-8)
                    )

                if dynamic_active:
                    output_gate = style_gate.to(
                        device=modified_output.device,
                        dtype=modified_output.dtype,
                    )
                    modified_output = (
                        output_gate * modified_output
                        + (1.0 - output_gate) * output
                    )

                if should_measure:
                    self.trajectory_diagnostics.append(
                        {
                            'layer': name,
                            'timestep': int(self.cur_t),
                            'denoise_progress': float(denoise_progress),
                            'dynamic_active': bool(dynamic_active),
                            'transport_active': bool(transport_active),
                            'token_count': int(q_cs.shape[1]),
                            'query_drift': float(drift.detach().cpu().item()),
                            'generation_weight': float(
                                generation_weight.float().mean().detach().cpu().item()
                            ),
                            'generation_weight_min': float(
                                weight_min.detach().cpu().item()
                            ),
                            'generation_weight_max': float(
                                weight_max.detach().cpu().item()
                            ),
                            'generation_weight_std': float(
                                weight_std.detach().cpu().item()
                            ),
                            'fraction_below_baseline': float(
                                fraction_below_baseline.detach().cpu().item()
                            ),
                            'fraction_above_baseline': float(
                                fraction_above_baseline.detach().cpu().item()
                            ),
                            'style_gate': float(style_gate.detach().cpu().item()),
                            'output_change_ratio': float(
                                output_change_ratio.detach().cpu().item()
                            ),
                            'transport_entropy': float(
                                transport_entropy.detach().cpu().item()
                            ),
                            'transport_confidence': float(
                                transport_confidence.detach().cpu().item()
                            ),
                        }
                    )

                return modified_output
        
        return hook

    def take_attn_features(self):
        features = self.attn_features
        self.attn_features = {name: {} for name in features}
        return features

    def reset_trajectory_diagnostics(self):
        self.trajectory_diagnostics = []
        self.trajectory_weight_ema = {}
        self.trajectory_transport_state = {}

    def take_trajectory_diagnostics(self):
        diagnostics = self.trajectory_diagnostics
        self.trajectory_diagnostics = []
        return diagnostics


def summarize_trajectory_diagnostics(entries, cfg):
    summary = {
        'trajectory_mode': cfg.trajectory_mode,
        'parameters': {
            'gamma': cfg.gamma,
            'drift_threshold': cfg.trajectory_drift_threshold,
            'min_generation_weight': cfg.trajectory_min_generation_weight,
            'max_generation_weight': cfg.trajectory_max_generation_weight,
            'guard_strength': cfg.trajectory_guard_strength,
            'min_style_gate': cfg.trajectory_min_style_gate,
            'start_fraction': cfg.trajectory_start_fraction,
            'end_fraction': cfg.trajectory_end_fraction,
            'token_temperature': cfg.trajectory_token_temperature,
            'temporal_smoothing': cfg.trajectory_temporal_smoothing,
            'output_strength': cfg.trajectory_output_strength,
            'transport_prototype_side': cfg.transport_prototype_side,
            'transport_temperature': cfg.transport_temperature,
            'transport_temporal_weight': cfg.transport_temporal_weight,
            'transport_velocity': cfg.transport_velocity,
            'transport_coverage_strength': cfg.transport_coverage_strength,
            'transport_sinkhorn_iterations': cfg.transport_sinkhorn_iterations,
            'transport_max_tokens': cfg.transport_max_tokens,
            'transport_min_tokens': cfg.transport_min_tokens,
            'transport_confidence_power': cfg.transport_confidence_power,
        },
        'entry_count': len(entries),
        'entries': entries,
    }
    if entries:
        for key in (
            'query_drift',
            'generation_weight',
            'generation_weight_min',
            'generation_weight_max',
            'generation_weight_std',
            'fraction_below_baseline',
            'fraction_above_baseline',
            'style_gate',
            'output_change_ratio',
            'transport_entropy',
            'transport_confidence',
        ):
            values = np.asarray([entry[key] for entry in entries], dtype=np.float64)
            summary[key] = {
                'mean': float(values.mean()),
                'minimum': float(values.min()),
                'maximum': float(values.max()),
            }
    return summary


def write_trajectory_diagnostics(entries, cfg, output_path):
    diagnostics_path = os.path.splitext(output_path)[0] + '.trajectory.json'
    summary = summarize_trajectory_diagnostics(entries, cfg)
    with open(diagnostics_path, 'w', encoding='utf-8') as diagnostics_file:
        json.dump(summary, diagnostics_file, ensure_ascii=False, indent=2)
    return diagnostics_path
    
    
if __name__ == "__main__":

    cfg = get_args()

    if cfg.resolution <= 0 or cfg.resolution % 8 != 0:
        raise ValueError("--resolution must be a positive multiple of 8")
    if not 0.0 <= cfg.gamma <= 1.0:
        raise ValueError("--gamma must be between 0 and 1")
    if cfg.T <= 0:
        raise ValueError("--T must be greater than 0")
    if cfg.trajectory_drift_threshold <= 0:
        raise ValueError("--trajectory_drift_threshold must be greater than 0")
    base_generation_weight = 1.0 - cfg.gamma
    if (
        cfg.trajectory_mode in (
            'dynamic', 'token_dynamic', 'output_fusion', 'transport'
        )
        and not 0.0
        <= cfg.trajectory_min_generation_weight
        <= base_generation_weight
        <= cfg.trajectory_max_generation_weight
        <= 1.0
    ):
        raise ValueError(
            "trajectory generation weights must satisfy 0 <= min <= "
            f"{base_generation_weight:.4f} <= max <= 1"
        )
    if cfg.trajectory_guard_strength < 0:
        raise ValueError("--trajectory_guard_strength must be non-negative")
    if not 0.0 <= cfg.trajectory_min_style_gate <= 1.0:
        raise ValueError("--trajectory_min_style_gate must be between 0 and 1")
    if not (
        0.0
        <= cfg.trajectory_start_fraction
        <= cfg.trajectory_end_fraction
        <= 1.0
    ):
        raise ValueError(
            "trajectory fractions must satisfy 0 <= start <= end <= 1"
        )
    if cfg.trajectory_token_temperature <= 0:
        raise ValueError("--trajectory_token_temperature must be greater than 0")
    if not 0.0 <= cfg.trajectory_temporal_smoothing < 1.0:
        raise ValueError(
            "--trajectory_temporal_smoothing must satisfy 0 <= value < 1"
        )
    if cfg.trajectory_output_strength < 0.0:
        raise ValueError("--trajectory_output_strength must be non-negative")
    if cfg.transport_prototype_side <= 0:
        raise ValueError("--transport_prototype_side must be greater than 0")
    if cfg.transport_temperature <= 0.0:
        raise ValueError("--transport_temperature must be greater than 0")
    if not 0.0 <= cfg.transport_temporal_weight <= 1.0:
        raise ValueError("--transport_temporal_weight must be between 0 and 1")
    if cfg.transport_velocity < 0.0:
        raise ValueError("--transport_velocity must be non-negative")
    if not 0.0 <= cfg.transport_coverage_strength <= 1.0:
        raise ValueError("--transport_coverage_strength must be between 0 and 1")
    if cfg.transport_sinkhorn_iterations < 0:
        raise ValueError("--transport_sinkhorn_iterations must be non-negative")
    if cfg.transport_max_tokens <= 0:
        raise ValueError("--transport_max_tokens must be greater than 0")
    if not 0 < cfg.transport_min_tokens <= cfg.transport_max_tokens:
        raise ValueError(
            "transport token limits must satisfy 0 < min_tokens <= max_tokens"
        )
    if cfg.transport_confidence_power < 0.0:
        raise ValueError("--transport_confidence_power must be non-negative")

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
        'trajectory_mode': cfg.trajectory_mode,
        'trajectory_drift_threshold': cfg.trajectory_drift_threshold,
        'trajectory_min_generation_weight': cfg.trajectory_min_generation_weight,
        'trajectory_max_generation_weight': cfg.trajectory_max_generation_weight,
        'trajectory_guard_strength': cfg.trajectory_guard_strength,
        'trajectory_min_style_gate': cfg.trajectory_min_style_gate,
        'trajectory_start_fraction': cfg.trajectory_start_fraction,
        'trajectory_end_fraction': cfg.trajectory_end_fraction,
        'trajectory_token_temperature': cfg.trajectory_token_temperature,
        'trajectory_temporal_smoothing': cfg.trajectory_temporal_smoothing,
        'trajectory_output_strength': cfg.trajectory_output_strength,
        'transport_prototype_side': cfg.transport_prototype_side,
        'transport_temperature': cfg.transport_temperature,
        'transport_temporal_weight': cfg.transport_temporal_weight,
        'transport_velocity': cfg.transport_velocity,
        'transport_coverage_strength': cfg.transport_coverage_strength,
        'transport_sinkhorn_iterations': cfg.transport_sinkhorn_iterations,
        'transport_max_tokens': cfg.transport_max_tokens,
        'transport_min_tokens': cfg.transport_min_tokens,
        'transport_confidence_power': cfg.transport_confidence_power,
        'save_trajectory_diagnostics': cfg.save_trajectory_diagnostics,
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
    unet_wrapper.reset_trajectory_diagnostics()
    images, latents = unet_wrapper.reverse_process(latent_cs, denoise_kwargs=denoise_kwargs) # reverse process save activations such as attn, res
    trajectory_diagnostics = unet_wrapper.take_trajectory_diagnostics()
    
    # save image
    images = [denormalize(image)[0] for image in images]
    image_last = images[-1]
    if cfg.save_intermediates:
        save_image(np.concatenate(images, axis=1), os.path.join(save_dir, "reverse_stylized.jpg"))
    if os.path.basename(cfg.output_name) != cfg.output_name:
        raise ValueError("--output_name must be a filename, not a path")
    output_path = os.path.join(save_dir, cfg.output_name)
    save_image(image_last, output_path)

    if cfg.save_trajectory_diagnostics:
        trajectory_path = write_trajectory_diagnostics(
            trajectory_diagnostics, cfg, output_path
        )
        print(f"Saved trajectory diagnostics to {os.path.abspath(trajectory_path)}")

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
    
