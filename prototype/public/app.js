const $ = id => document.getElementById(id);
let token = '', selected = null, report = null, polling = false;
const format = (x, places = 1) => Number.isFinite(x) ? x.toFixed(places) : '—';
async function api(path) {
  const response = await fetch(path, {headers: {Authorization: `Bearer ${token}`}, cache: 'no-store'});
  if (!response.ok) throw new Error(response.status === 401 ? 'Read token rejected.' : `API error ${response.status}`);
  return response.json();
}
function draw(history) {
  const svg = $('chart'), ns = 'http://www.w3.org/2000/svg';
  svg.replaceChildren();
  const add = (kind, attrs, label) => { const el = document.createElementNS(ns, kind);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
    if (label !== undefined) el.textContent = label; svg.append(el); return el; };
  for (const v of [0, 40, 60, 100]) {
    const y = 210 - v * 1.8;
    add('line', {x1:40, x2:700, y1:y, y2:y, stroke:'#d5dcce', 'stroke-dasharray':v === 40 ? '4 5' : 'none'});
    add('text', {x:8, y:y+4}, String(v));
  }
  const rows = history.filter(r => Number.isFinite(r.raw_validity_pct));
  if (!rows.length) return;
  const points = rows.map((r, i) => [40 + i * 650 / Math.max(1, rows.length-1), 210 - r.raw_validity_pct*1.8]);
  add('polyline', {points:points.map(p=>p.join(',')).join(' '), fill:'none', stroke:'#426b4a', 'stroke-width':3});
  points.forEach(([cx,cy],i) => {const dot = add('circle',{cx,cy,r:4,fill:'#17362f'});
    const title = document.createElementNS(ns,'title'); title.textContent=`${rows[i].stage} ${rows[i].iteration}: ${rows[i].raw_validity_pct.toFixed(1)}%`;dot.append(title);});
  $('chart-caption').textContent = `${rows.length} measured checkpoints. Dashed line: 40% PPO entry gate. Final evaluation uses a fresh sample.`;
}
async function show(id) {
  selected = id; report = await api(`/api/runs/${id}`);
  $('run-title').textContent = `Seed ${report.seed} / ${report.stage}`;
  $('run-kind').textContent = `${report.mode.toUpperCase()} / VALIDITY-ONLY`;
  $('run-status').textContent = report.status.toUpperCase().replaceAll('_',' ');
  $('warning').textContent = report.mode === 'smoke' ? 'SMOKE TEST: small fixture data, not evidence of molecular learning or drug-discovery performance.' : 'RESEARCH RUN: structural validity only. No binding, safety, synthesis or experimental validation claim.';
  $('validity').textContent = format(report.metrics.raw_validity_pct) + (Number.isFinite(report.metrics.raw_validity_pct) ? '%' : '');
  $('kl').textContent = format(report.metrics.prior_kl, 4);
  $('reward').textContent = format(report.metrics.mean_reward, 3);
  $('limitations').textContent = report.limitations.join('. ') + (report.error ? `. Failure: ${report.error}` : '');
  $('provenance').textContent = `${report.run_id} | ${report.dataset?.source || 'Preparing corpus'} | source ${report.provenance?.source_sha256?.slice(0,12) || 'unknown'} | base commit ${report.provenance?.commit?.slice(0,12) || 'unknown'} | received ${report.received_at}`;
  $('download').disabled = false;
  document.querySelectorAll('.run').forEach(el => el.classList.toggle('active', el.dataset.id === id));
  draw(report.history);
}
async function refresh() {
  if (!token || polling) return;
  polling = true;
  try {
    let all = [], cursor = null;
    do {const page = await api('/api/runs' + (cursor ? `?cursor=${encodeURIComponent(cursor)}` : ''));
      all.push(...page.runs); cursor = page.cursor;
    } while (cursor && all.length < 500);
    all.sort((a,b)=>b.received_at.localeCompare(a.received_at));
    $('run-count').textContent = `${all.length} runs`;
    $('runs').replaceChildren();
    for (const r of all) {
      const button = document.createElement('button');button.className='run';button.dataset.id=r.run_id;
      const title=document.createElement('b');title.textContent=`Seed ${r.seed} / ${r.mode}`;
      const detail=document.createElement('small');detail.textContent=`${r.status.replaceAll('_',' ')} · ${format(r.raw_validity_pct)}% valid`;
      button.append(title,detail);button.addEventListener('click',()=>show(r.run_id).catch(e=>$('connection-status').textContent=e.message));$('runs').append(button);
    }
    if (all.length) await show(selected && all.some(r=>r.run_id === selected) ? selected : all[0].run_id);
    else $('runs').textContent='No reports yet. Run the Colab smoke test to publish the first record.';
    $('connection-status').textContent=`Connected · refreshed ${new Date().toLocaleTimeString()}. Storage may take a short time to synchronize.`;
  } catch(e) { $('connection-status').textContent=e.message; }
  finally {polling=false;}
}
$('connect').addEventListener('submit',event=>{event.preventDefault();token=$('token').value.trim();$('token').value='';refresh();});
$('download').addEventListener('click',()=>{if(!report)return;const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`${report.run_id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
draw([]);
setInterval(()=>{if(!document.hidden)refresh();},15000);
