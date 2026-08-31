import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ---------- scene / camera / renderer ----------
const wrap = document.getElementById('canvas-wrap');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f1220);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
camera.position.set(5.6, 4.6, 6.4);
camera.lookAt(0, 0, 0);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
wrap.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08; controls.enablePan = false;
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dir = new THREE.DirectionalLight(0xffffff, 0.9); dir.position.set(5, 8, 6); scene.add(dir);
scene.add(new THREE.DirectionalLight(0xffffff, 0.35).translateX(-5));

// ---------- cube ----------
const CUBIE = 0.95;
const COLORS = { U:0xf7f7f7, D:0xffd500, R:0xc41e3a, L:0xff8c00, F:0x009e60, B:0x0051ba, I:0x1a1a1a };
const LETTER_HEX = { U:"#f7f7f7", R:"#c41e3a", F:"#009e60", D:"#ffd500", L:"#ff8c00", B:"#0051ba" };
const AXIS = { x:0, y:1, z:2 };
const AXIS_NAME = ['x','y','z'];
const cube = new THREE.Group();
scene.add(cube);
const cubies = [];
function faceColor(pos, axis, sign) {
  if (Math.sign(pos[axis]) === sign) { const l=[['R','L'],['U','D'],['F','B']][axis][sign<0?1:0]; return COLORS[l]; }
  return COLORS.I;
}
function makeMaterials(pos) {
  const spec=[[AXIS.x,+1],[AXIS.x,-1],[AXIS.y,+1],[AXIS.y,-1],[AXIS.z,+1],[AXIS.z,-1]];
  return spec.map(([a,s])=>new THREE.MeshStandardMaterial({color:faceColor(pos,a,s),roughness:.6,metalness:.05}));
}
function buildCube() {
  const geo=new THREE.BoxGeometry(CUBIE,CUBIE,CUBIE);
  for(let x=-1;x<=1;x++)for(let y=-1;y<=1;y++)for(let z=-1;z<=1;z++){
    if(x===0&&y===0&&z===0)continue;
    const mesh=new THREE.Mesh(geo,makeMaterials([x,y,z])); mesh.position.set(x,y,z); cube.add(mesh); cubies.push(mesh);
  }
}
function clearCube(){ while(cube.children.length) cube.remove(cube.children[0]); cubies.length=0; }
function rebuildSolved(){ clearCube(); buildCube(); }
buildCube();

// ---------- moves ----------
const D90=Math.PI/2;
function def(axisKey,layer,angSign){ return {axis:AXIS[axisKey],layer,angle:angSign*D90}; }
const MOVES_MAP={
  U:def('y',1,-1),"U'":def('y',1,1), D:def('y',-1,1),"D'":def('y',-1,-1),
  R:def('x',1,-1),"R'":def('x',1,1), L:def('x',-1,1),"L'":def('x',-1,-1),
  F:def('z',1,-1),"F'":def('z',1,1), B:def('z',-1,1),"B'":def('z',-1,-1),
};
function animateMove(name,duration){
  const m=MOVES_MAP[name]; if(!m)return Promise.resolve();
  return new Promise((resolve)=>{
    const pivot=new THREE.Group(); cube.add(pivot);
    const ax=m.axis,layer=m.layer,aname=AXIS_NAME[ax];
    const layerCubies=cubies.filter(c=>Math.round(c.position.getComponent(ax))===layer);
    layerCubies.forEach(c=>pivot.attach(c));
    const startRot=pivot.rotation[aname],endRot=startRot+m.angle,t0=performance.now();
    function frame(now){ const t=Math.min(1,(now-t0)/duration); pivot.rotation[aname]=startRot+(endRot-startRot)*t;
      if(t<1){requestAnimationFrame(frame);return;}
      pivot.rotation[aname]=endRot;
      layerCubies.forEach(c=>{cube.attach(c);c.position.set(Math.round(c.position.x),Math.round(c.position.y),Math.round(c.position.z));});
      cube.remove(pivot); resolve();
    }
    requestAnimationFrame(frame);
  });
}
async function applySequence(names,duration){ for(const n of names) await animateMove(n,duration); }

// ---------- kociemba order sticker index (matches backend) ----------
const FACE_OFF={U:0,R:9,F:18,D:27,L:36,B:45};
function stickerIndex(pos,n){
  const [x,y,z]=pos; let face,idx;
  if(n[1]===1){face='U';idx=(z+1)*3+(x+1);}
  else if(n[1]===-1){face='D';idx=(1-z)*3+(x+1);}
  else if(n[0]===1){face='R';idx=(1-y)*3+(z+1);}
  else if(n[0]===-1){face='L';idx=(1-y)*3+(z+1);}
  else if(n[2]===1){face='F';idx=(1-y)*3+(x+1);}
  else{face='B';idx=(1-y)*3+(1-x);}
  return FACE_OFF[face]+idx;
}
function makeMaterialsFromFacelet(pos,facelet){
  const spec=[[AXIS.x,+1],[AXIS.x,-1],[AXIS.y,+1],[AXIS.y,-1],[AXIS.z,+1],[AXIS.z,-1]];
  return spec.map(([a,s])=>{
    let color;
    if(Math.sign(pos[a])===s){ color=LETTER_HEX[facelet[stickerIndex(pos,[a===0?s:0,a===1?s:0,a===2?s:0])]]; }
    else color='#1a1a1a';
    return new THREE.MeshStandardMaterial({color,roughness:.6,metalness:.05});
  });
}
function rebuildFromFacelet(facelet){
  clearCube();
  const geo=new THREE.BoxGeometry(CUBIE,CUBIE,CUBIE);
  for(let x=-1;x<=1;x++)for(let y=-1;y<=1;y++)for(let z=-1;z<=1;z++){
    if(x===0&&y===0&&z===0)continue;
    const mesh=new THREE.Mesh(geo,makeMaterialsFromFacelet([x,y,z],facelet)); mesh.position.set(x,y,z); cube.add(mesh); cubies.push(mesh);
  }
}

// build a solved cube where each face is colored by its CENTER color (from the custom facelet)
function makeMaterialsFromColorMap(pos,colorMap){
  const spec=[[AXIS.x,+1],[AXIS.x,-1],[AXIS.y,+1],[AXIS.y,-1],[AXIS.z,+1],[AXIS.z,-1]];
  return spec.map(([a,s])=>{
    let color;
    if(Math.sign(pos[a])===s){ const l=[['R','L'],['U','D'],['F','B']][a][s<0?1:0]; color=colorMap[l]; }
    else color='#1a1a1a';
    return new THREE.MeshStandardMaterial({color,roughness:.6,metalness:.05});
  });
}
function rebuildCenterCube(centerColors){
  clearCube();
  const geo=new THREE.BoxGeometry(CUBIE,CUBIE,CUBIE);
  for(let x=-1;x<=1;x++)for(let y=-1;y<=1;y++)for(let z=-1;z<=1;z++){
    if(x===0&&y===0&&z===0)continue;
    const mesh=new THREE.Mesh(geo,makeMaterialsFromColorMap([x,y,z],centerColors)); mesh.position.set(x,y,z); cube.add(mesh); cubies.push(mesh);
  }
}
function centersFromFacelet(s){
  // kociemba order U,R,F,D,L,B, center index = offset+4
  const idx={U:4,R:13,F:22,D:31,L:40,B:49};
  const cc={}; for(const k in idx) cc[k]=LETTER_HEX[s[idx[k]]]; return cc;
}

// ---------- UI helpers ----------
const statusEl=document.getElementById('status');
const beamEl=document.getElementById('beam');
const scEl=document.getElementById('scramble-speed'), svEl=document.getElementById('solve-speed');
const scVal=document.getElementById('scramble-speed-val'), svVal=document.getElementById('solve-speed-val');
const btnScramble=document.getElementById('btn-scramble');
const btnSolve=document.getElementById('btn-solve');
const btnReset=document.getElementById('btn-reset');
const moveBtns=Array.from(document.querySelectorAll('[data-move]'));
const btnModeNormal=document.getElementById('mode-normal');
const btnModeCustom=document.getElementById('mode-custom');
const btnCustom=document.getElementById('btn-custom');
const customPanel=document.getElementById('custom-panel');
const faceletResult=document.getElementById('facelet-result');
const palette=Array.from(document.querySelectorAll('.swatch'));
const btnValidate=document.getElementById('btn-validate');
const btnSolveFacelet=document.getElementById('btn-solve-facelet');
const btnPreview=document.getElementById('btn-preview');
const btnCloseCustom=document.getElementById('btn-close-custom');
function setStatus(txt,ok=true){ statusEl.textContent=txt; statusEl.style.color=ok?'#cdd3e8':'#ff8080'; }
function scMs(){return parseInt(scEl.value,10);}
function svMs(){return parseInt(svEl.value,10);}
function sec(v){return (v/1000).toFixed(1);}
scEl.addEventListener('input',()=>scVal.textContent=sec(scMs()));
svEl.addEventListener('input',()=>svVal.textContent=sec(svMs()));

let scrambleMoves=[];
let busy=false;
let mode='normal';
function applyLock(){
  const normal=mode==='normal';
  btnScramble.disabled=busy||!normal; btnSolve.disabled=busy||!normal; btnReset.disabled=busy||!normal;
  moveBtns.forEach(x=>x.disabled=busy||!normal);
  btnModeNormal.disabled=busy; btnModeCustom.disabled=busy; btnCustom.disabled=busy;
}
function setBusy(b,excludeMove=false){ busy=b; applyLock(); }
function setMode(m){ mode=m; const normal=mode==='normal'; customPanel.classList.toggle('hidden',normal); btnModeNormal.classList.toggle('active',normal); btnModeCustom.classList.toggle('active',!normal); if(!normal&&!netBuilt) buildNet(); applyLock(); }
btnModeNormal.addEventListener('click',()=>{ if(!busy) setMode('normal'); });
btnModeCustom.addEventListener('click',()=>{ if(!busy) setMode('custom'); });
btnCustom.addEventListener('click',()=>{ if(!busy) setMode('custom'); });
btnCloseCustom.addEventListener('click',()=>{ setMode('normal'); });

function randomScramble(n=22){ const faces=['U','D','L','R','F','B']; const out=[]; let last=null;
  for(let i=0;i<n;i++){ let f; do{f=faces[Math.floor(Math.random()*faces.length)];}while(f===last); out.push(f+(Math.random()<0.5?"'":'')); last=f; } return out; }

btnScramble.addEventListener('click',async()=>{
  if(busy)return; setBusy(true);
  if(scrambleMoves.length){ const rev=scrambleMoves.slice().reverse().map(m=>m.endsWith("'")?m.slice(0,-1):m+"'"); await applySequence(rev,60); }
  scrambleMoves=randomScramble(22);
  setStatus(`打乱中（${scrambleMoves.length} 步）...`); await applySequence(scrambleMoves,scMs());
  setStatus(`已打乱 ${scrambleMoves.length} 步。点击“复原”让 AI 求解。`); setBusy(false);
});
moveBtns.forEach(b=>b.addEventListener('click',async()=>{
  if(busy)return; setBusy(true,true);
  await animateMove(b.dataset.move,scMs()); scrambleMoves.push(b.dataset.move);
  setStatus(`已手动转动 ${b.dataset.move}（累计 ${scrambleMoves.length} 步）。`); setBusy(false,true);
}));
btnReset.addEventListener('click',async()=>{
  if(busy)return; setBusy(true);
  const rev=scrambleMoves.slice().reverse().map(m=>m.endsWith("'")?m.slice(0,-1):m+"'");
  await applySequence(rev,60); scrambleMoves=[]; setStatus('已重置为复原态。'); setBusy(false);
});
btnSolve.addEventListener('click',async()=>{
  if(busy)return;
  if(scrambleMoves.length===0){ setStatus('当前已是复原态。',false); return; }
  setBusy(true); const beam=parseInt(beamEl.value,10);
  setStatus(`正在请求 AI 求解（beam=${beam}）...`); let resp;
  try{ const r=await fetch('/api/solve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scramble:scrambleMoves.join(' '),beam})}); resp=await r.json(); }
  catch(e){ setStatus('无法连接后端，请先启动 server.py。',false); setBusy(false); return; }
  if(resp.error){ setStatus(resp.error,false); setBusy(false); return; }
  if(!resp.solved){ setStatus('后端返回：未找到解。',false); setBusy(false); return; }
  setStatus(`AI 找到 ${resp.length} 步解法，正在自动演示...`); await applySequence(resp.moves,svMs());
  scrambleMoves=[]; setStatus(`✅ 已复原！共 ${resp.length} 步（beam=${beam}）。`); setBusy(false);
});

// ---------- custom facelet editor ----------
let selectedColor='U'; let netBuilt=false;
function buildNet(){
  if(netBuilt)return;
  const net=document.querySelector('.net');
  const faces=['U','R','F','D','L','B'];
  const layout={U:'1/2',R:'2/3',F:'2/2',D:'3/2',L:'2/1',B:'2/4'};
  faces.forEach(f=>{
    const fd=document.createElement('div'); fd.className='face'; fd.dataset.face=f;
    const [r,c]=layout[f].split('/'); fd.style.gridRow=r; fd.style.gridColumn=c; net.appendChild(fd);
    for(let p=0;p<9;p++){
      const cell=document.createElement('button'); cell.className='cell'; cell.dataset.face=f; cell.dataset.pos=p; cell.dataset.color=f; cell.style.background=LETTER_HEX[f];
      cell.addEventListener('click',()=>{ cell.dataset.color=selectedColor; cell.style.background=LETTER_HEX[selectedColor]; });
      fd.appendChild(cell);
    }
  });
  netBuilt=true;
}
palette.forEach(sw=>sw.addEventListener('click',()=>{ selectedColor=sw.dataset.color; palette.forEach(x=>x.classList.toggle('sel',x===sw)); }));
function buildFacelet(){ const cells=document.querySelectorAll('.cell'); const map={}; cells.forEach(c=>map[c.dataset.face+c.dataset.pos]=c.dataset.color);
  const faces=['U','R','F','D','L','B']; let s=''; faces.forEach(f=>{for(let p=0;p<9;p++)s+=map[f+p];}); return s; }
function validateFacelet(s){ const cnt={}; for(const ch of s)cnt[ch]=(cnt[ch]||0)+1; return ['U','R','F','D','L','B'].every(c=>cnt[c]===9); }
btnPreview.addEventListener('click',()=>{
  const s=buildFacelet();
  if(!validateFacelet(s)){ faceletResult.textContent='❌ 颜色数不合法，先修正再预览。'; faceletResult.style.color='#ff8080'; return; }
  rebuildFromFacelet(s);
  faceletResult.textContent='3D 预览已刷新（按你涂的盘面渲染）。'; faceletResult.style.color='#cdd3e8';
});
btnValidate.addEventListener('click',()=>{ const s=buildFacelet(); const ok=validateFacelet(s);
  faceletResult.textContent=ok?'✅ 每种颜色各 9 个，表现合法。':'❌ 颜色数不合法（每种颜色应各 9 个）。'; faceletResult.style.color=ok?'#7fe08a':'#ff8080'; });
btnSolveFacelet.addEventListener('click',async()=>{
  const s=buildFacelet(); if(!validateFacelet(s)){ faceletResult.textContent='❌ 颜色数不合法，请先修正。'; faceletResult.style.color='#ff8080'; return; }
  const beam=parseInt(beamEl.value,10);
  faceletResult.textContent=`请求求解中（beam=${beam}）...`; faceletResult.style.color='#cdd3e8'; let resp;
  try{ const r=await fetch('/api/solve_facelet',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({facelet:s,beam})}); resp=await r.json(); }
  catch(e){ faceletResult.textContent='无法连接后端。'; faceletResult.style.color='#ff8080'; return; }
  if(resp.error){ faceletResult.textContent='❌ '+resp.error; faceletResult.style.color='#ff8080'; return; }
  // exact 3D reconstruction: solved colored by centers -> apply inverse(solution) -> animate solution
  rebuildCenterCube(centersFromFacelet(s));
  const inv=resp.moves.slice().reverse().map(m=>m.endsWith("'")?m.slice(0,-1):m+"'");
  await applySequence(inv,60);
  faceletResult.textContent=`解法 ${resp.length} 步，正在 3D 演示...`; faceletResult.style.color='#cdd3e8';
  await applySequence(resp.moves,svMs());
  faceletResult.textContent=`✅ 已复原！解法（${resp.length} 步）：${resp.moves.join(' ')}`; faceletResult.style.color='#7fe08a';
});

// ---------- render loop ----------
function resize(){ const w=wrap.clientWidth,h=wrap.clientHeight; renderer.setSize(w,h); camera.aspect=w/h; camera.updateProjectionMatrix(); }
new ResizeObserver(resize).observe(wrap); resize();
(function loop(){ requestAnimationFrame(loop); controls.update(); renderer.render(scene,camera); })();