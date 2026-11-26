# Training-free three-trajectory alignment

The original StyleID injection uses a fixed query mixture at every denoising
step:

```text
Q_base = gamma * Q_content + (1 - gamma) * Q_generated
O_base = Attention(Q_base, K_style, V_style)
```

The dynamic mode keeps the same content and style inversion trajectories, but
uses the current generation trajectory to adapt the mixture. Query drift is
the mean cosine distance between the generated and content queries. When drift
is small, the generated query receives more weight so that style matching is
conditioned on the actual generation state. When drift reaches the threshold,
the method falls back to the original StyleID query mixture rather than
reinforcing an unreliable generated state. An optional content guard can also
reduce style injection and blend back the generation branch's natural
self-attention output; it is disabled by default.

```text
drift = mean(1 - cosine(Q_generated, Q_content))
content_confidence = 1 - clamp(drift / drift_threshold, 0, 1)
generation_weight = lerp(min_generation_weight, max_generation_weight,
                         content_confidence)
Q_tri = (1 - generation_weight) * Q_content
        + generation_weight * Q_generated
style_gate = clamp(exp(-guard_strength * drift), min_style_gate, 1)
O_tri = style_gate * Attention(Q_tri, K_style, V_style)
        + (1 - style_gate) * Attention(Q_generated, K_generated, V_generated)
```

No model parameter is trained or updated.

## Token-wise dynamic alignment

The `token_dynamic` mode replaces the single layer-level controller with a
spatial controller. For every attention head and query token it measures
cosine drift, estimates robust per-head quartiles, and maps the drift to a
generation-query weight. The mapping is calibrated so that the median token
keeps StyleID's original `1 - gamma` weight. Tokens with below-median drift use
more of the current generation query, while high-drift tokens fall back toward
the content inversion query.

```text
drift_i = 1 - cosine(Q_generated_i, Q_content_i)
scale = max(quantile_75(drift) - quantile_25(drift), 0.02)
logit_i = logit((base_weight - min_weight) / (max_weight - min_weight))
          + (median(drift) - drift_i) / (scale * token_temperature)
weight_i = min_weight + (max_weight - min_weight) * sigmoid(logit_i)
Q_token_i = (1 - weight_i) * Q_content_i + weight_i * Q_generated_i
```

An optional exponential moving average of the token weights provides
training-free inter-step smoothing. It is reset for every generated image.
The implementation still uses PyTorch SDPA and does not materialize an
attention matrix, so the 256-pixel smoke run retained the 2.05 GiB peak CUDA
tensor-memory footprint.

By default, dynamic alignment is active only for normalized denoising progress
0.2 through 0.8. The early high-noise stage and the late detail-reconstruction
stage use the exact StyleID baseline operation.

The default minimum and maximum generation weights straddle StyleID's fixed
`1 - gamma` weight. High-confidence states receive more generated-query weight;
low-confidence states receive more content-query weight. At the midpoint the
operation matches the baseline query mixture.

## Single-pair runs

Baseline:

```powershell
.\run_styleid.ps1 -Content 00.png -Style 00.png `
  -SaveDir results\trajectory_baseline `
  -TrajectoryMode baseline -SaveTrajectoryDiagnostics
```

Dynamic alignment:

```powershell
.\run_styleid.ps1 -Content 00.png -Style 00.png `
  -SaveDir results\trajectory_dynamic `
  -TrajectoryMode dynamic -SaveTrajectoryDiagnostics
```

Token-wise dynamic alignment:

```powershell
.\run_styleid.ps1 -Content 00.png -Style 00.png `
  -SaveDir results\trajectory_token_dynamic `
  -TrajectoryMode token_dynamic `
  -TrajectoryMinGenerationWeight 0.05 `
  -TrajectoryMaxGenerationWeight 0.55 `
  -TrajectoryTemporalSmoothing 0.5 `
  -SaveTrajectoryDiagnostics
```

## Paired comparison

Run `run_styleid_batch.py` twice with identical inputs and sampling settings,
changing only `--trajectory_mode`. Then aggregate paired metrics:

```powershell
.\.venv\Scripts\python.exe `
  diffusers_implementation\compare_trajectory_results.py `
  --baseline_dir results\trajectory_baseline `
  --dynamic_dir results\trajectory_dynamic `
  --output_dir results\trajectory_comparison
```

Every run can emit a `.trajectory.json` file containing per-layer and
per-timestep query drift, generation weight, and style gate. These diagnostics
test the mechanism directly rather than relying only on final-image metrics.

## Initial validation

The checked-in implementation was evaluated without training on 25 paired
samples (5 content images x 5 style images), using SD 1.5, 512 x 512 images,
20 DDIM steps, gamma 0.75, temperature 1.5, and layers 7 through 11. The
existing `results/all_default` outputs were used as the baseline after fresh
baseline runs were verified to be byte-identical to them.

| Aggregate metric | Baseline | Dynamic | Delta |
|---|---:|---:|---:|
| Content score | 0.6278 | 0.6278 | -0.0001 |
| Style score | 0.7079 | 0.7080 | +0.0001 |
| Balanced score | 0.6623 | 0.6623 | +0.0000 |
| Style color similarity | 0.6434 | 0.6441 | +0.0007 |

Across 2,500 recorded layer/timestep entries, mean query drift was 0.1969.
The controller was active for 1,500 entries, with a mean generated-query
weight of 0.2334 versus StyleID's fixed 0.25. The method is therefore stable
and genuinely responds to trajectory drift, but this query-weight-only
prototype does not yet provide a meaningful aggregate improvement. A stronger
follow-up should update an explicit generated-to-style correspondence matrix
or constrain inter-step trajectory velocity rather than only changing the
query mixture.

## Token-wise validation

The token-wise controller was evaluated on the same 25 pairs and settings as
the initial validation. Its active token weights had a mean of 0.2463, close
to the 0.25 StyleID baseline, but a nonzero mean spatial standard deviation of
0.0716 and an observed range of 0.0500 to 0.5381. Thus the controller changes
local correspondence without globally biasing every location in the same
direction.

| Aggregate metric | Baseline | Global dynamic | Token dynamic | Token delta |
|---|---:|---:|---:|---:|
| Content structure score | 0.9301 | 0.9299 | 0.9304 | +0.0004 |
| Content gray SSIM | 0.4053 | 0.4053 | 0.4063 | +0.0009 |
| Content edge F1 | 0.5481 | 0.5481 | 0.5496 | +0.0015 |
| Content score | 0.6278 | 0.6278 | 0.6288 | +0.0010 |
| Style score | 0.7079 | 0.7080 | 0.7071 | -0.0008 |
| Balanced score | 0.6623 | 0.6623 | 0.6625 | +0.0002 |

Token dynamic improved the aggregate content score on 23 of 25 pairs and edge
F1 on 22 of 25 pairs. It therefore fixes the global controller's inability to
react locally, but the gain remains small and comes with a slight style-score
reduction. The next method should align the generated-to-style attention
distribution directly instead of further widening the query weights.

## Prototype correspondence transport

The `transport` mode directly changes which style values are read instead of
only perturbing the query. Style keys and values are spatially pooled into a
small prototype grid. Content-to-style and generation-to-style correspondence
maps are computed at every active denoising step:

```text
A_cs(t) = softmax(cosine(Q_content(t), K_style_proto(t)) / temperature)
A_gs(t) = softmax(cosine(Q_generated(t), K_style_proto(t)) / temperature)
```

The previous transport map is advanced by the change of the content inversion
correspondence. The predicted map, current content map, and current generation
map are fused in log probability space. A few Sinkhorn row/column
normalizations encourage broader use of style prototypes. The resulting map
directly transports pooled style values:

```text
log A_pred(t) = log A_transport(t-1)
                + velocity * (log A_cs(t) - log A_cs(t-1))
log A_fused(t) = w_current/2 * log A_cs(t)
                 + w_current/2 * log A_gs(t)
                 + w_temporal * log A_pred(t)
A_transport(t) = Sinkhorn(exp(log A_fused(t)))
O_transport(t) = A_transport(t) V_style_proto(t)
```

A JS-divergence confidence gate compares `A_cs` and `A_gs` per token. Regions
where the two trajectories disagree fall back toward the original StyleID
output. This preserves an exact baseline path and avoids materializing a full
4096 x 4096 attention matrix.

The final tested configuration applies transport only to 4096-token
high-resolution layers, uses a 16 x 16 style prototype grid, and keeps the
1024-token layers unchanged:

```powershell
.\.venv\Scripts\python.exe diffusers_implementation\run_styleid_batch.py `
  --trajectory_mode transport `
  --trajectory_output_strength 0.2 `
  --transport_prototype_side 16 `
  --transport_temperature 0.07 `
  --transport_temporal_weight 0.25 `
  --transport_velocity 0.5 `
  --transport_coverage_strength 0.3 `
  --transport_sinkhorn_iterations 3 `
  --transport_min_tokens 4096 `
  --transport_max_tokens 4096 `
  --transport_confidence_power 1.0 `
  --trajectory_start_fraction 0.2 `
  --trajectory_end_fraction 0.8
```

## Correspondence transport validation

All full comparisons use the same 25 pairs, SD 1.5 checkpoint, 512 x 512
resolution, 20 DDIM steps, gamma 0.75, temperature 1.5, and injection layers
7 through 11.

| Method | Content score | Style score | Balanced score | Edge F1 | Mean output change |
|---|---:|---:|---:|---:|---:|
| StyleID baseline | 0.6278 | 0.7079 | 0.6623 | 0.5481 | 0.0% |
| Global query dynamic | 0.6278 | 0.7080 | 0.6623 | 0.5481 | negligible |
| Token query dynamic | 0.6288 | 0.7071 | 0.6625 | 0.5496 | small |
| Transport, high resolution, strength 0.1 | 0.6292 | 0.7068 | 0.6626 | 0.5515 | 5.6% |
| Transport, high resolution, strength 0.2 | 0.6303 | 0.7054 | 0.6627 | 0.5541 | 11.2% |
| Transport + JS confidence gate | 0.6303 | 0.7057 | 0.6628 | 0.5538 | 9.9% |

For the final confidence-gated configuration, content score improved on 18 of
25 pairs, edge F1 on 20 of 25, and balanced score on 16 of 25. The mean
content score gain was +0.0025 and the mean edge-F1 gain was +0.0058, while
style score decreased by 0.0022. Thus explicit correspondence transport has a
substantially larger and measurable effect than query interpolation, but it
still exposes a content/style trade-off. Content image 03 consistently
degraded, so a future controller should estimate sample-level transport risk
instead of relying only on local correspondence confidence.
