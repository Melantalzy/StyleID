"""Post-hoc texture association with existing StyleID output metrics."""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from experiment_lpips_content import fetch_groups, read
from run_styleid_diffusers import load_input_image
from analyze_styleid_rating_followup import write_csv


def rank(a):
    order = np.argsort(a, kind='stable')
    sorted_a = a[order]
    _, start, count = np.unique(sorted_a, return_index=True, return_counts=True)
    result = np.empty(len(a), dtype=float)
    result[order] = np.repeat(start + (count - 1) / 2, count)
    return result


def rho(x, y):
    a, b = rank(np.asarray(x)), rank(np.asarray(y))
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / denom) if denom else float('nan')


def main():
    rows = fetch_groups()
    alex = {(r['dataset'], r['content'], r['style']): r for r in read('results/experiment_alexnet_gram/pair_scores.csv')}
    lpips = {(r['dataset'], r['content'], r['style']): r for r in read('results/experiment_lpips_content/pair_scores.csv')}
    cache = {}
    for row in rows:
        ds, content, style = row['dataset'], row['content'], row['style']
        key = (ds, content, style)
        if (ds, style) not in cache:
            if ds == 'original': path = Path('data/sty') / f'{style}.png'
            elif ds == 'data_vis': path = Path('data_vis/sty') / f'{style}.png'
            else: path = Path('downloads/style-transfer-dataset/styles') / f'style_{style}.jpg'
            rgb = load_input_image(str(path), 512).astype(np.float32) / 255
            gray = rgb @ np.array([.2126, .7152, .0722], dtype=np.float32)
            cache[(ds, style)] = {f'texture_{s}': float(np.abs(gray - cv2.GaussianBlur(gray, (0, 0), s, borderType=cv2.BORDER_REFLECT_101)).mean()) for s in (1, 2, 4, 8)}
        row.update(cache[(ds, style)])
        metrics_path = Path(row['on_output']).with_suffix('.metrics.json')
        metrics = json.loads(metrics_path.read_text(encoding='utf8'))
        for field in ('style_gram_similarity', 'style_patch_similarity', 'style_color_similarity', 'style_score', 'content_ssim_gray', 'content_structure_score'):
            row[field] = float(metrics[field])
        row['alex_gram_on'] = float(alex[key]['alex_gram_on'])
        row['lpips_on'] = float(lpips[key]['lpips_on'])
    metrics = ('style_gram_similarity', 'style_patch_similarity', 'style_color_similarity', 'style_score', 'alex_gram_on', 'content_ssim_gray', 'content_structure_score', 'lpips_on')
    rng = np.random.default_rng(4701)
    summary, per_content = [], []
    for ds in ('original', 'data_vis', 'artflow_new_styles'):
        sub = [r for r in rows if r['dataset'] == ds]
        contents, styles = sorted({r['content'] for r in sub}), sorted({r['style'] for r in sub})
        lookup = {(r['content'], r['style']): r for r in sub}
        assert len(lookup) == len(contents) * len(styles), f'Nonrectangular dataset {ds}'
        for sigma in (1, 2, 4, 8):
            x = np.array([lookup[(contents[0], s)][f'texture_{sigma}'] for s in styles])
            for metric in metrics:
                y = np.array([[lookup[(c, s)][metric] for s in styles] for c in contents])
                values = np.array([rho(x, yy) for yy in y])
                for c, val in zip(contents, values):
                    per_content.append({'dataset': ds, 'content': c, 'sigma': sigma, 'metric': metric, 'rho': val})
                boot = []
                for _ in range(500):
                    ci = rng.integers(0, len(contents), len(contents))
                    si = rng.integers(0, len(styles), len(styles))
                    v = [rho(x[si], y[i, si]) for i in ci]
                    boot.append(float(np.nanmean(v)))
                lo, hi = np.quantile(boot, [.025, .975])
                summary.append({'dataset': ds, 'sigma': sigma, 'metric': metric, 'contents': len(contents), 'styles': len(styles), 'mean_rho': float(values.mean()), 'positive': int(sum(values > 0)), 'negative': int(sum(values < 0)), 'ci_low': float(lo), 'ci_high': float(hi)})
        print(f'Finished {ds}: {len(contents)} contents x {len(styles)} styles', flush=True)
    out = Path('results/experiment_texture_dataset')
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / 'cases.csv', rows, list(rows[0]))
    write_csv(out / 'per_content.csv', per_content, list(per_content[0]))
    write_csv(out / 'summary.csv', summary, list(summary[0]))
    (out / 'manifest.json').write_text(json.dumps({'type': 'post-hoc exploratory associations, no causal inference', 'feature': 'mean absolute luma residual after Gaussian blur, pixels scaled to [0,1], RGB 512 input preprocessing', 'sigmas': [1,2,4,8], 'ci': '500 crossed content/style bootstrap replicates, percentile 95%, exploratory', 'limitations': 'auto metrics not validated quality measures; styles shared between datasets; pre-existing data, multiplicity and metric coupling; JPEG LPIPS/AlexNet, other scores from run records'}, indent=2), encoding='utf8')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
