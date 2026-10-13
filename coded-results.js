/* Coded InverseMandala v2: site shows the frozen CI receipt, not a browser rerun. */
(async () => {
  const $ = id => document.getElementById(id);
  const labels = {
    single: 'One N=16 mandala',
    duplicate_control: 'Three duplicate N=16 views',
    center: 'Three shifted-center views',
    coded: 'N=16 + two coded masks',
    contrast: 'Normal + inverse contrast + shifted center',
    symmetry: 'Symmetries 3, 7 and 16',
    max_min: 'Max + contrast inverse (two views)'
  };
  const order = ['single','duplicate_control','center','coded','contrast','symmetry','max_min'];
  let receipt;
  try {
    const res = await fetch('results/coded_v2.json', {cache:'no-store'});
    if (!res.ok) throw new Error('HTTP ' + res.status);
    receipt = await res.json();
  } catch (err) {
    $('codedChart').textContent='The frozen JSON receipt could not be loaded here. Use a local HTTP server or open it on GitHub Pages.';
    return;
  }
  const scenePicker = $('codedScene');
  function draw() {
    const name = scenePicker.value;
    const data = name === 'all' ? receipt.mean_source_psnr :
      Object.fromEntries(Object.entries(receipt.scenes[name]).map(([k,v])=>[k,v.source_psnr]));
    const max = Math.max(...Object.values(data), 1);
    const chart = $('codedChart');chart.replaceChildren();
    order.forEach((method) => {
      const row = document.createElement('div'); row.className='coded-row';
      const label = document.createElement('span');label.className='coded-label';label.textContent=labels[method];
      const rail = document.createElement('div');rail.className='coded-rail';
      const bar = document.createElement('div');bar.className='coded-fill' + (method==='symmetry'?' highlight':'');
      bar.style.width=(100*data[method]/max).toFixed(2)+'%';
      rail.appendChild(bar);
      const value=document.createElement('strong');value.className='coded-value';value.textContent=data[method].toFixed(2)+' dB';
      row.append(label,rail,value);chart.appendChild(row);
    });
    const coded=receipt.collision.coded;
    $('codedWitness').textContent=coded.max_over_views.toFixed(4)+' / 1.0000';
    $('codedMean').textContent=data.coded.toFixed(2)+' dB';
    $('codedSym').textContent=data.symmetry.toFixed(2)+' dB';
    if(name==='all') {
      $('codedCoverage').textContent='Choose an individual scene';
    } else {
      const c=receipt.scenes[name].coded.winner_coverage;
      const s=receipt.scenes[name].symmetry.winner_coverage;
      $('codedCoverage').textContent=(100*c).toFixed(1)+'% vs '+(100*s).toFixed(1)+'%';
    }
    $('codedScope').textContent=name==='all'?'Average of three synthetic scenes · n=3, not a population estimate':
      'Fixed synthetic '+name+' scene · exact OpenCV rerender scored against hidden source';
  }
  scenePicker.addEventListener('change',draw);draw();
})();
