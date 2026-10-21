// Invoke the user's suggested MIT-licensed API with its checked-out worker.
// Dated financial statements use the separate Yahoo statement collector.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {gzipSync} from 'node:zlib';
import {pathToFileURL} from 'node:url';

const [apiCheckout, state, budgetText='160'] = process.argv.slice(2);
if (!apiCheckout || !state) throw Error('API checkout and state directory required');
const upstreamCommit='147cd5c92a7888104e0be63eef7ea72a5b61ad3e';
const source='https://github.com/0xramm/Indian-Stock-Market-API/tree/'+upstreamCommit;
const folder=path.join(state,'master_data','api_quote_inputs');fs.mkdirSync(folder,{recursive:true});
const rows=JSON.parse(fs.readFileSync(path.join(state,'master_data','api_public_universe.json'),'utf8'));
const journalPath=path.join(state,'master_data','indian_api_attempts.json');
const journal=fs.existsSync(journalPath)?JSON.parse(fs.readFileSync(journalPath,'utf8')):{};
const digest=s=>crypto.createHash('sha256').update(s).digest('hex');
rows.sort((a,b)=>(journal[a.isin]||'').localeCompare(journal[b.isin]||'')||digest(a.isin).localeCompare(digest(b.isin)));
const normal=s=>String(s||'').replace(/\b(?:limited|ltd)\.?\s*$/i,'').replaceAll('&','AND').toUpperCase().replaceAll(/[^A-Z0-9]/g,'');
const originalFetch=globalThis.fetch;let lastRequest=0;let denied=false;
globalThis.fetch=async (...args)=>{
  if(denied)throw Error('SOURCE_ACCESS_LIMIT');
  const delay=Math.max(0,lastRequest+400-Date.now());if(delay)await new Promise(r=>setTimeout(r,delay));lastRequest=Date.now();
  const r=await originalFetch(args[0],{...(args[1]||{}),signal:AbortSignal.timeout(20000)});
  if([403,429].includes(r.status)){denied=true;throw Error('SOURCE_ACCESS_LIMIT_HTTP_'+r.status);}
  return r;
};
const {default:api}=await import(pathToFileURL(path.join(apiCheckout,'src','index.js')));
let accepted=0,processed=0;const errors=[];
const started=Date.now();
for(const r of rows.slice(0,Number(budgetText))){
  if(denied || Date.now()-started>150000)break;
  try{
    const response=await api.fetch(new Request('http://localhost/stock?'+new URLSearchParams({symbol:r.nse_current_symbol+'.NS',res:'num'})));
    const raw=await response.text();const data=JSON.parse(raw);
    if(data.status!=='success'||data.ticker!==r.nse_current_symbol+'.NS'||data.exchange!=='NSE'||data.data?.currency!=='INR'||normal(data.data.company_name)!==normal(r.nse_current_name))throw Error('Quote identity/currency mismatch or unavailable');
    const hash=digest(raw);const archive=path.join(state,'master_data','raw',hash);fs.mkdirSync(archive,{recursive:true});fs.writeFileSync(path.join(archive,'INDIAN_STOCK_MARKET_API.json.gz'),gzipSync(raw));
    const receipt={symbol:r.nse_current_symbol,isin:r.isin,provider_family:'YAHOO_FINANCE',publisher:'INDIAN_STOCK_MARKET_API',upstream_repository:source,upstream_commit:upstreamCommit,
      first_observed_utc:new Date().toISOString(),source_sha256:hash,period_kind:'quote_snapshot',
      not_a_dated_financial_statement:true,independent_of_yahoo:false,data:data.data};
    fs.writeFileSync(path.join(folder,r.isin+'.json'),JSON.stringify(receipt,null,2));accepted++;
  }catch(e){errors.push({symbol:r.nse_current_symbol,error:String(e).slice(0,160)});}
  processed++;journal[r.isin]=new Date().toISOString();
  if(processed%25===0){fs.writeFileSync(journalPath,JSON.stringify(journal));console.log('Indian Stock Market API',processed,'accepted',accepted);}
}
fs.writeFileSync(journalPath,JSON.stringify(journal));
const summary={public_universe:rows.length,processed,accepted,errors,stopped_on_access_limit:denied,provider_family:'YAHOO_FINANCE',source,private_model_ranks_used_in_requests:false};
fs.writeFileSync(path.join(state,'master_data','indian_api_collection.json'),JSON.stringify(summary,null,2));console.log(JSON.stringify(summary));
