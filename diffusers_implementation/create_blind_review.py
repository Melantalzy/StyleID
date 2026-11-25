"""Prepare a local, blinded human-rating pilot for StyleID on/off outputs."""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from PIL import Image

from experiment_lpips_content import fetch_groups
from run_styleid_diffusers import load_input_image


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>风格迁移盲评</title><style>
body{font:16px/1.55 "Microsoft YaHei",sans-serif;max-width:1260px;margin:2em auto;padding:0 16px;background:#f5f7fa;color:#222}
.card{background:white;border:1px solid #d8dee8;border-radius:10px;padding:16px;margin:18px 0}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}img{width:100%;height:auto}h2{font-size:19px}
fieldset{margin:7px 0;border:1px solid #d8dee8;border-radius:6px}label{display:inline-block;margin:3px}
button{background:#225da8;color:white;border:0;padding:10px 18px;border-radius:6px;cursor:pointer}
input[type=text]{padding:6px} .muted{color:#596375}@media(max-width:700px){.grid{grid-template-columns:repeat(2,1fr)}}
</style></head><body><h1>风格迁移盲评试验</h1>
<p>每组展示内容图、风格图及两个随机排列的结果 A/B。独立评分每张结果：1 很差，3 一般，5 很好；各维度不要求两张图分数不同。</p>
<p>内容保持：主体与结构是否保留；风格相似：参考图的风格是否体现；无伪影：5 代表没有明显瑕疵，1 代表瑕疵严重。请结合两张参考图评分。</p>
<p>此页面在本地运行，不上传评价。进度保存在当前浏览器，完成后点“导出评分”保存 JSON；请将 JSON 文件交回用于分析。独立评分者请填写不同编号。</p>
<p>评分者编号：<input id="rater" type="text" placeholder="例如 R01"> <button id="export">导出评分</button> <button id="reset">清空当前进度</button> <span id="progress" class="muted"></span></p>
<main id="cases"></main>
<script>
const cases=CASES_JSON;
const key='styleid_blind_review_v1';
let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){saved={}}
const main=document.getElementById('cases');
const dims=[['content','内容保持'],['style','风格相似'],['artifact','无伪影']];
for(const item of cases){
  const card=document.createElement('section');card.className='card';
  const heading=document.createElement('h2');heading.textContent='样本 '+item.id;card.appendChild(heading);
  const grid=document.createElement('div');grid.className='grid';
  for(const [name,file] of [['内容图',item.content],['风格图',item.style],['结果 A',item.a],['结果 B',item.b]]){
    const col=document.createElement('div');const title=document.createElement('strong');title.textContent=name;col.appendChild(title);
    const img=document.createElement('img');img.src=file;img.alt=name+' '+item.id;col.appendChild(img);grid.appendChild(col);
  }card.appendChild(grid);
  for(const arm of ['A','B'])for(const [dim,title] of dims){
    const field=document.createElement('fieldset');const legend=document.createElement('legend');legend.textContent='结果 '+arm+' · '+title;field.appendChild(legend);
    const name=item.id+'_'+arm+'_'+dim;
    for(let score=1;score<=5;score++){const label=document.createElement('label');const radio=document.createElement('input');radio.type='radio';radio.name=name;radio.value=score;radio.checked=String(saved[name]||'')===String(score);radio.addEventListener('change',()=>{saved[name]=score;localStorage.setItem(key,JSON.stringify(saved));update()});label.appendChild(radio);label.append(' '+score);field.appendChild(label)}
    card.appendChild(field);
  }main.appendChild(card);
}
function update(){document.getElementById('progress').textContent='已评分 '+Object.keys(saved).length+' / '+(cases.length*6)+' 项'}update();
document.getElementById('export').addEventListener('click',()=>{
 const rater=document.getElementById('rater').value.trim();if(!rater){alert('请填写评分者编号');return}
 if(Object.keys(saved).length!==cases.length*6){alert('请完成全部评分后再导出');return}
 const blob=new Blob([JSON.stringify({version:'styleid_blind_review_v1',rater,scores:saved},null,2)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='styleid_blind_scores_'+rater+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
});
document.getElementById('reset').addEventListener('click',()=>{if(confirm('确认清空当前浏览器的评分进度？请先导出需要保留的评分。')){localStorage.removeItem(key);location.reload()}});
</script></body></html>'''


def main():
    out = Path("results/blind_review_pilot")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing review: {out}")
    out.mkdir(parents=True)
    assets = out / "images"
    assets.mkdir()
    rng = random.Random(71423)
    rows = fetch_groups()
    chosen = []
    for dataset in ("original", "data_vis", "artflow_new_styles"):
        candidates = [r for r in rows if r["dataset"] == dataset]
        rng.shuffle(candidates)
        seen_c, seen_s = set(), set()
        for row in candidates:
            if row["content"] in seen_c or row["style"] in seen_s:
                continue
            chosen.append(row)
            seen_c.add(row["content"])
            seen_s.add(row["style"])
            if len(seen_c) == 4:
                break
    if len(chosen) != 12:
        raise ValueError("Could not select 12 unique pilot cases")
    rng.shuffle(chosen)
    cases, key = [], []
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
        arms = [row["on_output"], row["off_output"]]
        rng.shuffle(arms)
        for arm, path in zip(("A", "B"), arms):
            shutil.copyfile(path, assets / f"{case}_{arm}.jpg")
        cases.append({"id": case, "content": f"images/{case}_content.png", "style": f"images/{case}_style.png",
                      "a": f"images/{case}_A.jpg", "b": f"images/{case}_B.jpg"})
        key.append({"id": case, "dataset": row["dataset"], "content": row["content"], "style": row["style"],
                    "input_patch_similarity": row["input_patch_similarity"],
                    "A": "on" if arms[0] == row["on_output"] else "off",
                    "B": "on" if arms[1] == row["on_output"] else "off"})
    (out / "index.html").write_text(HTML.replace("CASES_JSON", json.dumps(cases, ensure_ascii=False)), encoding="utf-8")
    (out / "answer_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Blinded review page: {(out / 'index.html').resolve()}")
    print("12 image pairs, randomized arms; answer key stored separately. No scores generated.")


if __name__ == "__main__":
    main()
