'use strict';
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const esc = v => String(v ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (n,d=1) => typeof n==='number' && Number.isFinite(n) ? n.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}) : '—';
const clock = seconds => new Date(seconds*1000).toLocaleTimeString('en-GB',{hour12:false});
const labels = {pending:'Queued / Prefill',decode:'Generating',completed:'Completed',failed:'Failed',cancelled:'Cancelled'};
const state = {requests:[],events:[],config:[],selected:null,detail:null,tab:'summary',view:'requests',busy:false,connected:false};
const api = path => new URL(`api/${path}`,location.href);

async function read(path,options={}) {
  const response = await fetch(api(path),{cache:'no-store',...options});
  if(!response.ok) throw Error(`${response.status}`);
  return response.json();
}
function filtered() {
  const search=$('#request-search').value.trim().toLowerCase(),status=$('#status-filter').value;
  return state.requests.filter(r=>(status==='all'||(status==='active'?['pending','decode'].includes(r.status):status===r.status)) && r.id.toLowerCase().includes(search));
}
function renderRequests() {
  const focused=document.activeElement?.dataset.focusRequest;
  const rows=filtered();
  $('#request-rows').innerHTML=rows.length?rows.map(r=>`<tr data-request="${r.key}" class="${state.selected===r.key?'selected':''}" aria-selected="${state.selected===r.key}"><td><button class="request-id" data-focus-request="${r.key}" aria-label="${esc(r.id)}" title="${esc(r.id)} · ${clock(r.created)}">${esc(r.id.length>16?r.id.slice(-12):r.id)}</button></td><td><span class="state ${r.status}">${labels[r.status]||esc(r.status)}</span></td><td class="numeric">${fmt(r.input,0)} <span class="muted">/</span> ${fmt(r.output,0)}</td><td class="numeric">${fmt(r.prefill_rate,0)}</td><td class="numeric rate">${fmt(r.decode_rate)}</td><td class="numeric">${fmt(r.gen_rate)}</td></tr>`).join(''):'<tr class="empty-row"><td colspan="6">No requests</td></tr>';
  $('#request-count').textContent=state.requests.length;
  $('#result-count').textContent=rows.length===state.requests.length?`${rows.length}`:`${rows.length} / ${state.requests.length}`;
  if(focused) $$('[data-focus-request]').find(e=>e.dataset.focusRequest===focused)?.focus({preventScroll:true});
}
function showView(name) {
  state.view=['configuration','events'].includes(name)?name:'requests';
  $$('.view').forEach(e=>{e.hidden=e.id!==`view-${state.view}`;});
  $$('[data-view]').forEach(e=>{const on=e.dataset.view===state.view;e.classList.toggle('selected',on);if(on)e.setAttribute('aria-current','page');else e.removeAttribute('aria-current');});
}
function fields(rows) {return `<dl class="detail-fields">${rows.map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl>`;}
function publicDetail() {
  if(!state.detail) return null;
  const result={...state.detail};
  if(!$('#show-content').checked){delete result.prompt;delete result.response;}
  return result;
}
function renderDetail() {
  const r=state.detail;
  $('#detail-pane').hidden=!r;
  $('#view-requests').classList.toggle('has-detail',!!r);
  if(!r)return;
  $('#detail-id').textContent=r.id;
  $$('[data-detail]').forEach(e=>{const on=e.dataset.detail===state.tab;e.setAttribute('aria-selected',String(on));e.tabIndex=on?0:-1;});
  $('#detail-content').setAttribute('aria-labelledby',`tab-${state.tab}`);
  const scroll=$('#detail-content').scrollTop;
  if(state.tab==='summary'){
    const seconds=n=>n===null?'—':`${fmt(n,2)} s`;
    $('#detail-content').innerHTML=`<div class="detail-summary">${fields([['Queue',seconds(r.queue)],['Prefill',seconds(r.prefill)],['Decode',seconds(r.decode)],['Total',seconds(r.total)]])}${fields([['Received',clock(r.created)],['TTFT',seconds(r.ttft)],['Finish reason',r.finish_reason||'—'],['Cached tokens',fmt(r.cached,0)]])}</div>${!r.single_sequence?'<p class="content-disabled">n &gt; 1: per-request rates unavailable</p>':''}${r.error?`<p class="detail-error">${esc(r.error)}</p>`:''}`;
  }else if(state.tab==='content'){
    $('#detail-content').innerHTML=$('#show-content').checked?`<div class="content-columns"><section><h3>Prompt</h3><pre>${esc(r.prompt||'No content')}</pre></section><section><h3>Response</h3><pre>${esc(r.response||'—')}</pre></section></div>`:'<span class="content-disabled">Request text hidden</span>';
  }else $('#detail-content').innerHTML=`<pre class="json-content">${esc(JSON.stringify(publicDetail(),null,2))}</pre>`;
  $('#detail-content').scrollTop=scroll;
}
async function selectRequest(key) {
  state.selected=key;state.tab='summary';renderRequests();
  try{const detail=await read(`requests/${key}`);if(state.selected===key){state.detail=detail;renderDetail();}}
  catch{if(state.selected===key){closeDetail();toast('Request no longer retained.');}}
}
function closeDetail(){const key=state.selected;state.selected=null;state.detail=null;renderRequests();renderDetail();$$('[data-focus-request]').find(e=>e.dataset.focusRequest===String(key))?.focus({preventScroll:true});}
function renderEvents(){
  const query=$('#event-search').value.toLowerCase(),level=$('#event-level').value;
  const rows=state.events.filter(e=>(level==='all'||level===e.level)&&e.message.toLowerCase().includes(query));
  $('#event-rows').innerHTML=rows.length?rows.map(e=>`<tr><td>${clock(e.created)}</td><td><span class="event-level ${e.level}">${esc(e.level)}</span></td><td>${e.key?`<button class="event-row-link" data-event-request="${e.key}">${esc(e.message)}</button>`:esc(e.message)}</td></tr>`).join(''):'<tr class="empty-row"><td colspan="3">No logs</td></tr>';
}
const connection=$('.status-bar>span:first-child');
async function poll(force=false){
  if(state.busy||(!force&&(!$('#auto-refresh').checked||document.hidden)))return;
  state.busy=true;
  try{
    const data=await read('state');state.connected=true;state.requests=data.requests;state.events=data.events;state.config=data.config;
    $('#engine-state').textContent=({ready:'Ready',error:'Error',stopped:'Stopped'})[data.status]||data.status;
    $('#engine-dot').classList.toggle('neutral',data.status!=='ready');
    $('#model-name').textContent=data.model;$('#model-name').title=data.model;
    $('#server-address').textContent=location.host;
    const profile=data.profile||'CLI';
    for(const selector of ['#profile-name','#config-profile']){const element=$(selector);element.textContent=profile;element.title=profile;}
    $('#kv-state').textContent=(data.kv_usage===null?'—':`${fmt(data.kv_usage*100)}%`)+(data.kv_budget_bytes?` / ${fmt(data.kv_budget_bytes/1e9)} GB`:'');
    $('#running-count').textContent=data.running??'—';$('#waiting-count').textContent=data.waiting??'—';
    $('.version').textContent=data.version?`v${data.version}`:'—';
    $('#config-rows').innerHTML=state.config.map(([k,v])=>`<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join('');
    renderRequests();renderEvents();
    if(state.selected!==null){try{const key=state.selected,detail=await read(`requests/${key}`);if(state.selected===key){state.detail=detail;if(state.tab!=='content'||window.getSelection()?.isCollapsed)renderDetail();}}catch{closeDetail();}}
    connection.textContent=`Connected · ${new Date().toLocaleTimeString('en-GB',{hour12:false})}`;
  }catch{
    state.connected=false;$('#engine-state').textContent='Disconnected';$('#engine-dot').classList.add('neutral');connection.textContent='Server disconnected';
  }finally{state.busy=false;}
}
function closeMenu(){$('#app-menu').hidden=true;$('#app-menu-toggle').setAttribute('aria-expanded','false');}
let toastTimer;
function toast(message){$('#toast').textContent=message;$('#toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>{$('#toast').hidden=true;},2200);}
document.addEventListener('click',e=>{
  const row=e.target.closest('[data-request]');if(row)selectRequest(Number(row.dataset.request));
  const tab=e.target.closest('[data-detail]');if(tab){state.tab=tab.dataset.detail;renderDetail();}
  const link=e.target.closest('[data-event-request]');if(link){location.hash='requests';showView('requests');selectRequest(Number(link.dataset.eventRequest));}
  if(!e.target.closest('.menu-anchor'))closeMenu();
});
$('#request-search').addEventListener('input',renderRequests);$('#status-filter').addEventListener('change',renderRequests);
$('#event-search').addEventListener('input',renderEvents);$('#event-level').addEventListener('change',renderEvents);
$('#detail-close').addEventListener('click',closeDetail);
$('#app-menu-toggle').addEventListener('click',()=>{const on=$('#app-menu').hidden;$('#app-menu').hidden=!on;$('#app-menu-toggle').setAttribute('aria-expanded',String(on));});
$('#auto-refresh').addEventListener('change',()=>{$('#footer-state').textContent=$('#auto-refresh').checked?'Auto refresh':'Refresh paused';if($('#auto-refresh').checked)poll(true);});
$('#show-content').addEventListener('change',renderDetail);
$('#copy-config').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(state.config.map(([k,v])=>`${k}: ${v}`).join('\n'));toast('Copied');}catch{toast('Could not copy');}});
$('#export-requests').addEventListener('click',()=>{const data={requests:filtered(),selected:publicDetail()};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='vllm-requests.json';document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);closeMenu();});
$('#tray-stop').addEventListener('click',()=>{closeMenu();$('#stop-dialog').showModal();});
$('#stop-cancel').addEventListener('click',()=>$('#stop-dialog').close());
$('#stop-confirm').addEventListener('click',async()=>{try{await read('stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirm:true})});$('#stop-dialog').close();toast('Stopping server');}catch{toast('Stop request failed');}});
document.addEventListener('keydown',e=>{
  if(e.key==='Escape'){if(!$('#app-menu').hidden)closeMenu();else if(state.selected!==null&&state.view==='requests')closeDetail();}
  if(e.target.matches('[data-detail]')&&['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){const tabs=['summary','content','json'];let i=tabs.indexOf(state.tab);i=e.key==='Home'?0:e.key==='End'?2:(i+(e.key==='ArrowRight'?1:2))%3;state.tab=tabs[i];renderDetail();$(`#tab-${state.tab}`).focus();e.preventDefault();}
  if(e.key==='/'&&!e.target.matches('input,textarea,select')&&!document.querySelector('dialog[open]')){e.preventDefault();location.hash='requests';showView('requests');$('#request-search').focus();}
});
window.addEventListener('hashchange',()=>showView(location.hash.slice(1)));
document.addEventListener('visibilitychange',()=>{if(!document.hidden)poll();});
showView(location.hash.slice(1));poll(true);setInterval(poll,1000);
