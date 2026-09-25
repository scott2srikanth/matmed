import test from 'node:test';
import assert from 'node:assert/strict';
import worker from '../src/worker.js';

const report = {schema_version:1, run_id:'matmed-42-test', seed:42, mode:'smoke',
  status:'completed', objective:'structural_validity_only', history:[{stage:'final',iteration:1,raw_validity_pct:25}],
  metrics:{raw_validity_pct:25,attempts:4,valid_count:1}};
function environment() {
  const data = new Map();
  return {READ_TOKEN:'reader', WRITE_TOKEN:'writer', RUNS:{
    async put(key,value,options){data.set(key,{value,metadata:options.metadata});},
    async get(key){return data.has(key) ? JSON.parse(data.get(key).value) : null;},
    async list(){return {keys:[...data].map(([name,v])=>({name,metadata:v.metadata})),list_complete:true};},
  }, ASSETS:{fetch:async()=>new Response('dashboard')}};
}
const request = (path, method='GET', token='reader', body) => new Request(`https://matmed.example${path}`, {
  method, headers:{Authorization:`Bearer ${token}`,'Content-Type':'application/json'},
  ...(body === undefined ? {} : {body:JSON.stringify(body)}),
});
test('upload, list and retrieve a real-shaped run report',async()=>{
  const env=environment();
  assert.equal((await worker.fetch(request('/api/runs/'+report.run_id,'PUT','writer',report),env)).status,200);
  const got=await (await worker.fetch(request('/api/runs/'+report.run_id),env)).json();
  assert.equal(got.metrics.raw_validity_pct,25);
  assert.equal((await (await worker.fetch(request('/api/runs'),env)).json()).runs.length,1);
});
test('separate read and write credentials and reject missing configuration',async()=>{
  const env=environment();
  assert.equal((await worker.fetch(request('/api/runs','GET','writer'),env)).status,401);
  assert.equal((await worker.fetch(request('/api/runs/'+report.run_id,'PUT','reader',report),env)).status,401);
  env.READ_TOKEN='';assert.equal((await worker.fetch(request('/api/runs'),env)).status,401);
});
test('reject invalid metrics, invalid JSON, oversized body and cross-origin',async()=>{
  const env=environment();
  for (const invalid of [null,{...report,metrics:{raw_validity_pct:101}}, {...report,history:[{loss:null}]}]) {
    assert.equal((await worker.fetch(request('/api/runs/'+report.run_id,'PUT','writer',invalid),env)).status,400);
  }
  const bad=request('/api/runs/'+report.run_id,'PUT','writer',{...report,padding:'x'.repeat(270000)});
  assert.equal((await worker.fetch(bad,env)).status,413);
  const req=request('/api/runs');req.headers.set('Origin','https://untrusted.example');
  assert.equal((await worker.fetch(req,env)).status,403);
});
test('same-origin CORS, missing reports, health and static dashboard',async()=>{
  const env=environment();
  const preflight=new Request('https://matmed.example/api/runs',{method:'OPTIONS',headers:{Origin:'https://matmed.example'}});
  assert.equal((await worker.fetch(preflight,env)).status,204);
  assert.equal((await worker.fetch(request('/api/runs/matmed-missing'),env)).status,404);
  assert.equal((await worker.fetch(request('/api/health','GET',''),env)).status,200);
  assert.equal(await (await worker.fetch(new Request('https://matmed.example/'),env)).text(),'dashboard');
});
