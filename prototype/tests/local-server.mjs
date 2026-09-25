// In-memory integration harness only; deployment uses Worker KV.
import http from 'node:http';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import worker from '../src/worker.js';
const root=fileURLToPath(new URL('../public/',import.meta.url));
const data=new Map();
const env={READ_TOKEN:'local-read',WRITE_TOKEN:'local-write',RUNS:{
  async put(k,v,o){data.set(k,{value:v,metadata:o.metadata});},
  async get(k){return data.has(k)?JSON.parse(data.get(k).value):null;},
  async list(){return {keys:[...data].map(([name,v])=>({name,metadata:v.metadata})),list_complete:true};},
},ASSETS:{async fetch(req){const pathname=new URL(req.url).pathname;
  const file=pathname==='/'?'index.html':pathname.slice(1);
  if(!['index.html','style.css','app.js'].includes(file))return new Response('Not found',{status:404});
  return new Response(await readFile(path.join(root,file)),{headers:{'Content-Type':
    file.endsWith('.html')?'text/html':file.endsWith('.css')?'text/css':'application/javascript'}});
}}};
http.createServer(async(req,res)=>{try{
  const chunks=[];for await(const c of req)chunks.push(c);
  const request=new Request(`http://127.0.0.1:${process.env.PORT||8787}${req.url}`,{method:req.method,
    headers:req.headers,...(['GET','HEAD'].includes(req.method)?{}:{body:Buffer.concat(chunks)})});
  const result=await worker.fetch(request,env);res.writeHead(result.status,Object.fromEntries(result.headers));
  res.end(Buffer.from(await result.arrayBuffer()));
}catch(e){res.writeHead(500);res.end(String(e));}}).listen(Number(process.env.PORT||8787),'127.0.0.1',()=>console.log('Local test server ready'));
