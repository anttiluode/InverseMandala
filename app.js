/* InverseMandala: dependency-free interactive explorer.
 * Mirrors the original 2mandalatools rotation/odd-flip/max construction,
 * except using continuous bilinear interpolation instead of OpenCV's 1/32 table.
 * The inverse solver NEVER inspects `source` or `winner`.
 */
(() => {
'use strict';
const S=96, P=S*S;
const $ = id => document.getElementById(id);
const els={scene:$('scene'),sym:$('sym'),views:$('views'),original:$('original'),mandala:$('mandala'),provenance:$('provenance'),reconstruction:$('reconstruction'),status:$('status'),solve:$('solve')};
let source=new Float32Array(P), current=null, candidate=null, attempt=0, custom=false, witnessVariant=false;
const cache=new Map();
function geometry(N){
  if(cache.has(N))return cache.get(N);
  const warps=[];
  for(let k=0;k<N;k++){
    const index=new Int32Array(P*4),weights=new Float32Array(P*4);
    index.fill(-1);
    const angle=2*Math.PI*k/N,c=Math.cos(angle),s=Math.sin(angle),cx=S>>1,cy=S>>1;
    for(let y=0;y<S;y++)for(let x=0;x<S;x++){
      const prex=(k&1)?S-1-x:x;
      const a=prex-cx,b=y-cy;
      const sx=c*a-s*b+cx,sy=s*a+c*b+cy;
      const left=Math.floor(sx),top=Math.floor(sy),fx=sx-left,fy=sy-top;
      const offsets=[[0,0,(1-fx)*(1-fy)],[1,0,fx*(1-fy)],[0,1,(1-fx)*fy],[1,1,fx*fy]];
      const base=(y*S+x)*4;
      for(let h=0;h<4;h++){
        const [ox,oy,w]=offsets[h],u=left+ox,v=top+oy;
        if(u>=0&&u<S&&v>=0&&v<S){index[base+h]=v*S+u;weights[base+h]=w;}
      }
    }
    warps.push({index,weights});
  }
  cache.set(N,warps);return warps;
}
function sample(warp,src,p){let a=0;const q=p*4;for(let j=0;j<4;j++){const i=warp.index[q+j];if(i>=0)a+=src[i]*warp.weights[q+j];}return a;}
function forward(src,N){
  const ws=geometry(N),out=new Float32Array(P),win=new Uint8Array(P);
  out.fill(-Infinity);
  for(let k=0;k<N;k++)for(let p=0;p<P;p++){
    const v=sample(ws[k],src,p);
    if(v>out[p]){out[p]=v;win[p]=k;}
  }
  return {out,win};
}
function grayScene(name){
  const off=document.createElement('canvas');off.width=off.height=S;
  const c=off.getContext('2d');
  const sx=S/64;
  c.fillStyle='#121212';c.fillRect(0,0,S,S);
  c.save();c.scale(sx,sx);
  if(name==='face'){
    c.fillStyle='#b3b3b3';c.strokeStyle='#eee';c.lineWidth=2;
    c.beginPath();c.ellipse(31,33,21,27,0,0,Math.PI*2);c.fill();c.stroke();
    c.fillStyle='#222';c.beginPath();c.ellipse(23,28,4,2.7,0,0,Math.PI*2);c.ellipse(39,27,4,2.3,0,0,Math.PI*2);c.fill();
    c.fillStyle='#fff';c.fillRect(22,27,2,2);
    c.strokeStyle='#292929';c.lineWidth=2;c.beginPath();c.arc(31,39,10,.05,Math.PI-.05);c.stroke();
    c.fillStyle='#eee';c.beginPath();c.arc(48,14,2,0,Math.PI*2);c.fill();
    c.strokeStyle='#343434';c.lineWidth=3;c.beginPath();c.moveTo(12,14);c.lineTo(22,7);c.lineTo(37,8);c.stroke();
  }else if(name==='symbols'){
    c.fillStyle='#ddd';c.fillRect(8,7,14,45);c.fillRect(14,7,25,10);
    c.fillStyle='#aaa';c.fillRect(17,26,26,12);c.beginPath();c.arc(47,49,11,0,Math.PI*2);c.fill();
    c.fillStyle='#fff';c.beginPath();c.moveTo(47,3);c.lineTo(57,19);c.lineTo(38,17);c.fill();
  }else if(name==='landscape'){
    c.fillStyle='#444';c.fillRect(0,41,64,23);
    c.fillStyle='#eee';c.beginPath();c.arc(48,13,9,0,Math.PI*2);c.fill();
    c.fillStyle='#999';c.beginPath();c.moveTo(0,48);c.lineTo(24,20);c.lineTo(45,48);c.fill();
    c.fillStyle='#ccc';c.beginPath();c.ellipse(15,34,11,9,0,0,Math.PI*2);c.fill();
    c.strokeStyle='#222';c.lineWidth=4;c.beginPath();c.moveTo(16,35);c.lineTo(15,51);c.stroke();
  }
  c.restore();
  const rgba=c.getImageData(0,0,S,S).data;
  const v=new Float32Array(P);for(let p=0;p<P;p++)v[p]=rgba[p*4]/255;
  return v;
}
function witness(){const out=new Float32Array(P).fill(1);if(witnessVariant)out[(S/2-12)*S+S/2+8]=0;return out;}
function updateSource(){source=els.scene.value==='witness'?witness():grayScene(els.scene.value);custom=false;render();}
function display(canvas,data,asProvenance=false,N=1){
  const ctx=canvas.getContext('2d'),img=ctx.createImageData(S,S);
  for(let p=0;p<P;p++){
    const v=Math.max(0,Math.min(255,Math.round((data[p]||0)*255))),j=p*4;
    if(!asProvenance)img.data[j]=img.data[j+1]=img.data[j+2]=v;
    else{
      const h=(data[p]/Math.max(1,N))*6,a=Math.floor(h),f=h-a;
      const q=(1-f)*210+35,t=f*210+35;
      const colors=[[245,t,35],[q,245,35],[35,245,t],[35,q,245],[t,35,245],[245,35,q]];
      const c=colors[Math.min(5,a)];img.data[j]=c[0];img.data[j+1]=c[1];img.data[j+2]=c[2];
    }
    img.data[j+3]=255;
  }
  ctx.putImageData(img,0,0);
}
function psnr(a,b){let mse=0;for(let i=0;i<P;i++){const d=a[i]-b[i];mse+=d*d;}mse/=P;return mse<1e-16?'∞':(-10*Math.log10(mse)).toFixed(1);}
function observedSet(){let ns=els.views.value==='current'?[Number(els.sym.value)]:els.views.value.split(',').map(Number);return ns.map(n=>({n,y:forward(source,n).out}));}
function render(){
  ++attempt;els.solve.disabled=false;els.solve.textContent='Try inverse ↗';
  const n=Number(els.sym.value);$('symval').textContent=n;current=forward(source,n);
  display(els.original,source);display(els.mandala,current.out);display(els.provenance,current.win,true,n);
  $('mandalaCaption').textContent=`Maximum of ${n} rotated / reflected copies`;
  $('provenanceBits').textContent=`${Math.ceil(Math.log2(n))} bits / pixel`;
  if(candidate){display(els.reconstruction,candidate);$('sourcePsnr').textContent=`${psnr(source,candidate)} dB`;$('forwardPsnr').textContent=`${psnr(current.out,forward(candidate,n).out)} dB`;}
  else{els.reconstruction.getContext('2d').clearRect(0,0,S,S);$('sourcePsnr').textContent='—';$('forwardPsnr').textContent='—';}
}
function setStatus(txt){els.status.textContent=txt;}
function computeLossAndGrad(x,obs){
  const grad=new Float64Array(P);let loss=0;
  for(const {n,y} of obs){
    const {out,win}=forward(x,n),warps=geometry(n);
    for(let p=0;p<P;p++){
      const d=out[p]-y[p];loss+=d*d/(obs.length*P);
      const w=warps[win[p]],q=p*4;
      for(let j=0;j<4;j++){
        const a=w.index[q+j];if(a>=0)grad[a]+=2*d*w.weights[q+j]/obs.length;
      }
    }
  }
  return {grad,loss};
}
async function solve(){
  if(els.solve.disabled)return;
  const epoch=++attempt;
  els.solve.disabled=true;els.solve.textContent='Solving…';
  const obs=observedSet();
  const x=new Float32Array(P);x.fill(1);
  for(let p=0;p<P;p++)for(const o of obs)x[p]=Math.min(x[p],o.y[p]);
  const m=new Float64Array(P),v=new Float64Array(P);
  let loss=0;
  for(let t=1;t<=100;t++){
    const r=computeLossAndGrad(x,obs);loss=r.loss;
    // Optional smoothness regularizer is local to the candidate, not the source.
    for(let row=0;row<S;row++)for(let col=0;col<S;col++){
      const p=row*S+col;
      if(col+1<S){const d=.003*(x[p]-x[p+1]);r.grad[p]+=d;r.grad[p+1]-=d;}
      if(row+1<S){const d=.003*(x[p]-x[p+S]);r.grad[p]+=d;r.grad[p+S]-=d;}
    }
    for(let p=0;p<P;p++){
      const g=r.grad[p];m[p]=.9*m[p]+.1*g;v[p]=.999*v[p]+.001*g*g;
      const step=.045*(m[p]/(1-Math.pow(.9,t)))/(Math.sqrt(v[p]/(1-Math.pow(.999,t)))+1e-8);
      x[p]=Math.max(0,Math.min(1,x[p]-step));
    }
    if(t%10===0){
      if(epoch!==attempt)return;
      candidate=x;display(els.reconstruction,candidate);
      $('sourcePsnr').textContent=psnr(source,candidate)+' dB';
      $('forwardPsnr').textContent=psnr(current.out,forward(candidate,Number(els.sym.value)).out)+' dB';
      setStatus(`Iteration ${t}/100 · forward objective ${loss.toFixed(5)} · Candidate reconstructed using ONLY ${obs.map(o=>'N='+o.n).join(', ')}. Source was not shown to the solver.`);
      await new Promise(resolve=>setTimeout(resolve,0));
    }
  }
  if(epoch!==attempt)return;
  candidate=x;els.solve.disabled=false;els.solve.textContent='Solve again ↗';
  setStatus(`Completed 100 iterations · ${obs.map(o=>'N='+o.n).join(', ')} · Forward agreement can be high even when actual source reconstruction is poor. The solver did not see the original or provenance.`);
}
$('scene').addEventListener('change',()=>{candidate=null;witnessVariant=false;updateSource();setStatus('Source changed. Run the inverse to see how much can be recovered.');});
els.sym.addEventListener('input',()=>{candidate=null;render();setStatus('Symmetry changed. The provenance colors show which rotated copy wins each pixel.');});
els.views.addEventListener('change',()=>{candidate=null;render();setStatus('Observation set changed. Compare one view against several views.');});
els.solve.addEventListener('click',solve);
$('uploadBtn').addEventListener('click',()=>$('file').click());
$('file').addEventListener('change',e=>{
  const file=e.target.files?.[0];if(!file)return;
  const url=URL.createObjectURL(file),img=new Image();
  img.onload=()=>{
    const off=document.createElement('canvas');off.width=off.height=S;const c=off.getContext('2d');
    const side=Math.min(img.width,img.height),dx=(img.width-side)/2,dy=(img.height-side)/2;
    c.drawImage(img,dx,dy,side,side,0,0,S,S);
    const a=c.getImageData(0,0,S,S).data;
    source=new Float32Array(P);for(let p=0;p<P;p++)source[p]=(.299*a[4*p]+.587*a[4*p+1]+.114*a[4*p+2])/255;
    candidate=null;custom=true;render();setStatus('Image loaded locally; center-cropped and converted to grayscale. Nothing was uploaded.');URL.revokeObjectURL(url);
  };
  img.onerror=()=>{setStatus('Image could not be loaded.');URL.revokeObjectURL(url);};
  img.src=url;
});
$('flipWitness').addEventListener('click',()=>{
  els.scene.value='witness';witnessVariant=!witnessVariant;source=witness();candidate=null;render();
  setStatus(`Witness source ${witnessVariant?'B (one dark pixel)':'A (all white)'}: symmetry 3/7/16 compositions are identical for these two different sources. Try switching back and forth.`);
  window.scrollTo({top:document.querySelector('.workbench').offsetTop-24,behavior:'smooth'});
});
updateSource();
})();
