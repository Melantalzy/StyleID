"""Create a second local human-rating pilot on 24 normal StyleID outputs."""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from PIL import Image

from experiment_lpips_content import fetch_groups
from run_styleid_diffusers import load_input_image


PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StyleID 风格迁移评分</title><style>
body{font:16px/1.6 "Microsoft YaHei",sans-serif;max-width:1100px;margin:2em auto;padding:0 16px;background:#f5f7fa;color:#222}
.card{background:#fff;border:1px solid #d8dee8;border-radius:10px;padding:16px;margin:18px 0}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}img{width:100%;height:auto}fieldset{margin:7px 0;border:1px solid #d8dee8;border-radius:6px}label{display:inline-block;margin:4px}button{background:#225da8;color:white;border:0;padding:10px 18px;border-radius:6px;cursor:pointer}input[type=text]{padding:6px}@media(max-width:700px){.grid{grid-template-columns:1fr 1fr}}
</style></head><body><h1>StyleID 结果评分</h1>
<p>请分别给每个结果的三个维度打分：1 很差、3 一般、5 很好。内容保持指主体与结构是否保留；风格相似指参考风格是否体现；无伪影中 5 表示没有明显瑕疵、1 表示瑕疵严重。</p>
<p>共 24 张结果。本页面只在本地保存进度，不上传数据。全部完成后请填写评分者编号并导出 JSON；多个评分者请使用不同浏览器配置，或在导出后清空当前进度。</p>
<p>评分者编号 <input id="rater" type="text" placeholder="例如 R01"> <button id="export">导出评分</button> <button id="reset">清空当前进度</button> <span id="progress"></span></p><main id="cases"></main>
<script>const cases=CASES_JSON;const key='styleid_rating_followup_v1';let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){saved={}}
const main=document.getElementById('cases');const dims=[['content','内容保持'],['style','风格相似'],['artifact','无伪影']];
for(const item of cases){const card=document.createElement('section');card.className='card';const heading=document.createElement('h2');heading.textContent='样本 '+item.id;card.appendChild(heading);const grid=document.createElement('div');grid.className='grid';
for(const [name,file] of [['内容图',item.content],['风格图',item.style],['迁移结果',item.result]]){const col=document.createElement('div');const title=document.createElement('strong');title.textContent=name;col.appendChild(title);const img=document.createElement('img');img.src=file;img.alt=name+' '+item.id;col.appendChild(img);grid.appendChild(col)}card.appendChild(grid);
for(const [dim,title] of dims){const field=document.createElement('fieldset');const legend=document.createElement('legend');legend.textContent=title;field.appendChild(legend);const name=item.id+'_'+dim;for(let n=1;n<=5;n++){const label=document.createElement('label');const radio=document.createElement('input');radio.type='radio';radio.name=name;radio.value=n;radio.checked=String(saved[name]||'')===String(n);radio.addEventListener('change',()=>{saved[name]=n;localStorage.setItem(key,JSON.stringify(saved));update()});label.appendChild(radio);label.append(' '+n);field.appendChild(label)}card.appendChild(field)}main.appendChild(card)}
function update(){document.getElementById('progress').textContent='已评分 '+Object.keys(saved).length+' / '+(cases.length*3)+' 项'}update();
document.getElementById('export').addEventListener('click',()=>{const rater=document.getElementById('rater').value.trim();if(!rater){alert('请填写评分者编号');return}if(Object.keys(saved).length!==cases.length*3){alert('请完成全部评分后再导出');return}const blob=new Blob([JSON.stringify({version:'styleid_rating_followup_v1',rater,scores:saved},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='styleid_followup_scores_'+rater+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)});
document.getElementById('reset').addEventListener('click',()=>{if(confirm('确认清空当前进度？请先导出需要保留的评分。')){localStorage.removeItem(key);location.reload()}});</script></body></html>'''


def main():
    out = Path("results/styleid_rating_followup")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing review: {out}")
    out.mkdir(parents=True)
    assets = out / "images"
    assets.mkdir()
    previous = json.loads(Path("results/blind_review_pilot/answer_key.json").read_text(encoding="utf-8"))
    excluded = {(r["dataset"], r["content"], r["style"]) for r in previous}
    rng = random.Random(890312)
    all_rows = fetch_groups()
    chosen = []
    for dataset in ("original", "data_vis", "artflow_new_styles"):
        candidates = [r for r in all_rows if r["dataset"] == dataset and (r["dataset"], r["content"], r["style"]) not in excluded]
        contents = sorted({r["content"] for r in candidates})
        selected_contents = rng.sample(contents, 2)
        for content in selected_contents:
            options = sorted((r for r in candidates if r["content"] == content), key=lambda r: r["input_patch_similarity"])
            for q in range(4):
                start, end = len(options) * q // 4, len(options) * (q + 1) // 4
                chosen.append(rng.choice(options[start:end]))
    if len(chosen) != 24:
        raise ValueError("Expected 24 outputs")
    rng.shuffle(chosen)
    cases, answer = [], []
    for index, row in enumerate(chosen, 1):
        case = f"{index:02d}"
        if row["dataset"] == "original":
            style_path = Path("data/sty") / f"{row['style']}.png"
        elif row["dataset"] == "data_vis":
            style_path = Path("data_vis/sty") / f"{row['style']}.png"
        else:
            style_path = Path("downloads/style-transfer-dataset/styles") / f"style_{row['style']}.jpg"
        Image.fromarray(load_input_image(row["content_path"], 512)).save(assets / f"{case}_content.png")
        Image.fromarray(load_input_image(str(style_path), 512)).save(assets / f"{case}_style.png")
        shutil.copyfile(row["on_output"], assets / f"{case}_result.jpg")
        cases.append({"id": case, "content": f"images/{case}_content.png", "style": f"images/{case}_style.png", "result": f"images/{case}_result.jpg"})
        answer.append({"id": case, "dataset": row["dataset"], "content": row["content"], "style": row["style"],
                       "input_patch_similarity": row["input_patch_similarity"], "output_path": row["on_output"]})
    (out / "index.html").write_text(PAGE.replace("CASES_JSON", json.dumps(cases, ensure_ascii=False)), encoding="utf-8")
    (out / "answer_key.json").write_text(json.dumps(answer, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rating page: {(out / 'index.html').resolve()}; 24 cases, 6 contents; no scores generated.")


if __name__ == "__main__":
    main()
