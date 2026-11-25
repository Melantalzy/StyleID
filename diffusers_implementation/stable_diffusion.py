
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline


MODEL_IDS = {
    "1.5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "2.0": "stabilityai/stable-diffusion-2-base",
    "2.1-base": "stabilityai/stable-diffusion-2-1-base",
    "2.1": "stabilityai/stable-diffusion-2-1",
}


# From "https://huggingface.co/blog/stable_diffusion"
def load_stable_diffusion(
    sd_version='1.5',
    precision_t=torch.float16,
    device="cuda",
    model_id=None,
    local_files_only=False,
):
    model_key = model_id or MODEL_IDS[sd_version]

    # The safety checker is not used by StyleID because the pipeline itself is
    # never called. Avoiding it saves both download size and GPU memory.
    load_kwargs = {
        "dtype": precision_t,
        "safety_checker": None,
        "feature_extractor": None,
        "requires_safety_checker": False,
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
    }
    if precision_t == torch.float16:
        load_kwargs["variant"] = "fp16"
    pipe = StableDiffusionPipeline.from_pretrained(model_key, **load_kwargs)
    
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    
    for module in (vae, text_encoder, unet):
        module.requires_grad_(False)
        module.eval()
        module.to(device)
    
    # import xformer
    # unet.enable_xformers_memory_efficient_attention()
    
    # Reuse the already downloaded scheduler configuration.
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    del pipe
    
    return vae, tokenizer, text_encoder, unet, scheduler

def decode_latent(latents, vae):
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    return image

def encode_latent(images, vae):
    # encode the image with vae
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.mode()
    latents = 0.18215 * latents
    return latents

def get_text_embedding(text, text_encoder, tokenizer, device="cuda"):
    # TODO currently, hard-coding for stable diffusion
    with torch.no_grad():

        prompt = [text]
        batch_size = len(prompt)
        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

        text_embeddings = text_encoder(text_input.input_ids.to(device))[0].to(device)
        max_length = text_input.input_ids.shape[-1]
        # print(max_length, text_input.input_ids)
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0].to(device)
    
    return text_embeddings, uncond_embeddings



def get_unet_layers(unet):
    
    layer_num = [i for i in range(12)]
    resnet_layers = []
    attn_layers = []
    
    for idx, ln in enumerate(layer_num):
        up_block_idx = idx // 3
        layer_idx = idx % 3
        
        resnet_layers.append(getattr(unet, 'up_blocks')[up_block_idx].resnets[layer_idx])
        if up_block_idx > 0:
            attn_layers.append(getattr(unet, 'up_blocks')[up_block_idx].attentions[layer_idx])
        else:
            attn_layers.append(None)
        
    return resnet_layers, attn_layers
        
        
        

# Diffusers attention code for getting query, key, value and attention map
def attention_op(
    attn,
    hidden_states,
    encoder_hidden_states=None,
    attention_mask=None,
    query=None,
    key=None,
    value=None,
    temperature=1.0,
    compute_output=True,
):
    residual = hidden_states
    
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, None)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    if query is None:
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    if key is None:
        key = attn.to_k(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
    if value is None:
        value = attn.to_v(encoder_hidden_states)
        value = attn.head_to_batch_dim(value)

    
    if key.shape[0] != query.shape[0]:
        key, value = key[:query.shape[0]], value[:query.shape[0]]

    if not compute_output:
        return None, query, key, value, None

    # PyTorch SDPA avoids materializing the full NxN attention matrix. This is
    # equivalent to scaling the query before the original attention operation.
    batch_heads, query_len, head_dim = query.shape
    batch_size = batch_heads // attn.heads
    query_4d = (query * temperature).view(batch_size, attn.heads, query_len, head_dim)
    key_4d = key.view(batch_size, attn.heads, key.shape[1], head_dim)
    value_4d = value.view(batch_size, attn.heads, value.shape[1], head_dim)
    hidden_states = F.scaled_dot_product_attention(
        query_4d,
        key_4d,
        value_4d,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
    )
    hidden_states = hidden_states.permute(0, 2, 1, 3).reshape(
        batch_size, query_len, attn.heads * head_dim
    )

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor
    
    return None, query, key, value, hidden_states
