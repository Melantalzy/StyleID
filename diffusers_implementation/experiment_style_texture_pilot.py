"""Exploratory multiscale style-image texture versus human ratings."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from analyze_styleid_rating_followup import spearman, write_csv


def texture(path):
    image = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255
    gray = image @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return {f'texture_sigma_{sigma}': float(np.abs(gray - cv2.GaussianBlur(
        gray, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT_101)).mean())
        for sigma in (1, 2, 4, 8)}


def rho(x, y):
    return spearman(x, y) if len(set(x)) > 1 and len(set(y)) > 1 else float('nan')


def main():
    base = Path('results/styleid_rating_followup')
    cases = json.loads((base / 'answer_key.json').read_text(encoding='utf8'))
    scores = json.loads(Path('C:/Users/Administrator/Downloads/styleid_followup_scores_R02.json').read_text(encoding='utf-8-sig'))
    if scores.get('version') != 'styleid_rating_followup_v1':
        raise ValueError('Incorrect scores version')
    expected = {f"{case['id']}_{d}" for case in cases for d in ('style', 'content', 'artifact')}
    if set(scores['scores']) != expected or any(type(v) is not int or not 1 <= v <= 5 for v in scores['scores'].values()):
        raise ValueError('Missing or invalid scores')
    rows = []
    for case in cases:
        row = {**case, 'rating_style': scores['scores'][case['id'] + '_style'],
               'rating_content': scores['scores'][case['id'] + '_content']}
        row.update(texture(base / 'images' / f"{case['id']}_style.png"))
        rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['content'])].append(row)
    per_content = []
    for (dataset, content), group in sorted(groups.items()):
        for sigma in (1, 2, 4, 8):
            field = f'texture_sigma_{sigma}'
            per_content.append({'dataset': dataset, 'content': content, 'sigma': sigma, 'n': len(group),
                                'rho_style': rho([r[field] for r in group], [r['rating_style'] for r in group])})
    rng = np.random.default_rng(1452)
    summary = []
    for sigma in (1, 2, 4, 8):
        field = f'texture_sigma_{sigma}'
        values = np.array([r['rho_style'] for r in per_content if r['sigma'] == sigma])
        observed = float(np.mean(values))
        boot = np.mean(rng.choice(values, size=(10000, len(values)), replace=True), axis=1)
        perm = []
        for _ in range(10000):
            perm.append(np.mean([rho([r[field] for r in group], rng.permutation([r['rating_style'] for r in group])) for group in groups.values()]))
        summary.append({'sigma_pixels': sigma, 'role': 'primary' if sigma == 2 else 'exploratory',
                        'mean_within_content_rho': observed, 'positive': int(sum(values > 0)),
                        'negative': int(sum(values < 0)), 'zero': int(sum(values == 0)),
                        'bootstrap_content_ci_low': float(np.percentile(boot, 2.5)),
                        'bootstrap_content_ci_high': float(np.percentile(boot, 97.5)),
                        'within_content_permutation_p_two_sided': (1 + sum(abs(x) >= abs(observed) - 1e-12 for x in perm)) / 10001})
    output = Path('results/experiment_style_texture_pilot')
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'cases.csv', rows, list(rows[0]))
    write_csv(output / 'per_content.csv', per_content, list(per_content[0]))
    write_csv(output / 'summary.csv', summary, list(summary[0]))
    (output / 'manifest.json').write_text(json.dumps({'method': '512px review style image, sRGB luma, mean absolute difference from Gaussian blur, reflect borders', 'sigmas_pixels': [1, 2, 4, 8], 'primary_sigma_pixels': 2, 'primary_outcome': 'human style rating', 'scores_excluded': 'artifact', 'sample': '24 preselected cases, 6 contents, 4 styles per content, one rater', 'bootstrap_caveat': 'content cluster bootstrap ignores repeated styles and has only 6 clusters', 'causality': 'observational, not causal'}, indent=2), encoding='utf8')
    print(f'Wrote {output}; 24 cases and six content groups')


if __name__ == '__main__':
    main()
