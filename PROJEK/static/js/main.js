/* ══════════════════════════════════════════
   CANVAS PARTICLES
══════════════════════════════════════════ */
(function () {
  const cv = document.getElementById('bgCanvas');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  let W, H, pts = [];
  const PAL = ['rgba(0,207,255,','rgba(124,94,246,','rgba(0,230,118,','rgba(240,248,255,'];
  function resize() { W = cv.width = window.innerWidth; H = cv.height = window.innerHeight; }
  function mkPt() {
    return {
      x:Math.random()*(W||1920), y:Math.random()*(H||1080),
      r:Math.random()*1.3+.2,
      vx:(Math.random()-.5)*.22, vy:(Math.random()-.5)*.22,
      a:Math.random()*.3+.05,
      col:PAL[Math.floor(Math.random()*PAL.length)]
    };
  }
  function tick() {
    ctx.clearRect(0,0,W,H);
    pts.forEach(p => {
      p.x+=p.vx; p.y+=p.vy;
      if(p.x<0)p.x=W; else if(p.x>W)p.x=0;
      if(p.y<0)p.y=H; else if(p.y>H)p.y=0;
      ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      ctx.fillStyle=p.col+p.a+')'; ctx.fill();
    });
    requestAnimationFrame(tick);
  }
  window.addEventListener('resize', resize);
  resize();
  pts = Array.from({length:65}, mkPt);
  tick();
})();

/* ══════════════════════════════════════════
   STATE
══════════════════════════════════════════ */
const DEFAULT_SETTINGS = { svm:true, rf:true, bert:true, fusion:'stacking' };

// Sentiment color lookup (module-level for reuse)
const SENT_COL = { Positive:'#00e676', Negative:'#ff4560', Neutral:'#ffbf00' };

let settings = (() => {
  try { return JSON.parse(localStorage.getItem('sentimentai_cfg')) || {...DEFAULT_SETTINGS}; }
  catch { return {...DEFAULT_SETTINGS}; }
})();

const history = []; // in-memory, clears on page close

/* ══════════════════════════════════════════
   PAGE NAVIGATION
══════════════════════════════════════════ */
const PAGE_TITLES = { analyze:'Analyze', history:'History', settings:'Settings' };

function navigate(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-'+page).classList.add('active');
  document.getElementById('nav-'+page).classList.add('active');
  document.getElementById('topbarTitle').textContent = PAGE_TITLES[page];

  if (page === 'history') renderHistory();
  if (page === 'settings') loadSettingsUI();
}

/* ══════════════════════════════════════════
   SETTINGS UI
══════════════════════════════════════════ */
function loadSettingsUI() {
  document.getElementById('toggleSVM').checked  = settings.svm;
  document.getElementById('toggleRF').checked   = settings.rf;
  document.getElementById('toggleBERT').checked = settings.bert;
  updateModelCards();
  selectFusionUI(settings.fusion);
}

function onModelToggle() {
  settings.svm  = document.getElementById('toggleSVM').checked;
  settings.rf   = document.getElementById('toggleRF').checked;
  settings.bert = document.getElementById('toggleBERT').checked;
  updateModelCards();
  checkFusionConflict();
}

function updateModelCards() {
  const map = { SVM: settings.svm, RF: settings.rf, BERT: settings.bert };
  Object.entries(map).forEach(([k,v]) => {
    document.getElementById('selCard'+k)?.classList.toggle('enabled', v);
    document.getElementById('selCard'+k)?.classList.toggle('disabled', !v);
  });
  updateChips();
}

function selectFusion(f) {
  settings.fusion = f;
  selectFusionUI(f);
  checkFusionConflict();
  updateChips();
}

function selectFusionUI(f) {
  ['stacking','average','vote'].forEach(id => {
    const el = document.getElementById('fusion'+cap(id));
    if(!el) return;
    el.classList.toggle('selected', id === f);
  });
}

function checkFusionConflict() {
  const activeCount = [settings.svm, settings.rf, settings.bert].filter(Boolean).length;
  const needsAll = settings.fusion === 'stacking' && activeCount < 3;
  const warn = document.getElementById('fusionWarn');
  if (warn) warn.classList.toggle('visible', needsAll);

  // Dim stacking option if not all 3
  const stackEl = document.getElementById('fusionStacking');
  if (stackEl) stackEl.classList.toggle('disabled', activeCount < 3 && settings.fusion !== 'stacking');
}

function cap(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

function saveSettings() {
  settings.svm    = document.getElementById('toggleSVM').checked;
  settings.rf     = document.getElementById('toggleRF').checked;
  settings.bert   = document.getElementById('toggleBERT').checked;
  try { localStorage.setItem('sentimentai_cfg', JSON.stringify(settings)); } catch {}
  updateChips();
  const msg = document.getElementById('saveMsg');
  if (msg) { msg.classList.add('show'); setTimeout(() => msg.classList.remove('show'), 2500); }
}

function resetSettings() {
  settings = {...DEFAULT_SETTINGS};
  loadSettingsUI();
  try { localStorage.removeItem('sentimentai_cfg'); } catch {}
  updateChips();
}

function updateChips() {
  // Topbar chips
  ['SVM','RF','BERT'].forEach(m => {
    const k = m.toLowerCase();
    const tb = document.getElementById('tb'+m);
    const sb = document.getElementById('sbChip'+m);
    if (tb) tb.classList.toggle('on', settings[k]), tb.classList.toggle('off', !settings[k]);
    if (sb) sb.classList.toggle('on', settings[k]), sb.classList.toggle('off', !settings[k]);
  });
  const fusionLabel = { stacking:'Stacking', average:'Avg Prob', vote:'Maj. Vote' };
  const lbl = fusionLabel[settings.fusion] || settings.fusion;
  document.getElementById('tbFusion').textContent  = lbl;
  document.getElementById('sbFusion').textContent  = lbl;
}

// Init chips on load
updateChips();

/* ══════════════════════════════════════════
   TEXTAREA
══════════════════════════════════════════ */
const textarea  = document.getElementById('reviewInput');
const charCount = document.getElementById('charCount');
textarea.addEventListener('input', () => {
  const n = textarea.value.length;
  charCount.textContent = `${n.toLocaleString()} / 5,000`;
  charCount.classList.toggle('warn',   n > 3000 && n <= 4500);
  charCount.classList.toggle('danger', n > 4500);
});
textarea.addEventListener('keydown', e => { if(e.ctrlKey && e.key==='Enter') analyzeReview(); });

/* ══════════════════════════════════════════
   ANALYZE
══════════════════════════════════════════ */
async function analyzeReview() {
  const text = textarea.value.trim();
  if (!text) { showError('Please enter a review text to analyze.'); return; }

  const activeModels = ['svm','rf','bert'].filter(m => settings[m]);
  if (!activeModels.length) {
    showError('No models selected. Please enable at least one model in Settings.');
    navigate('settings');
    return;
  }

  const btn     = document.getElementById('analyzeBtn');
  const btnText = document.getElementById('btnText');
  const btnIcon = document.getElementById('btnIcon');
  const spinner = document.getElementById('btnSpinner');

  document.getElementById('resultsWrapper').style.display = 'none';
  clearError();

  btn.disabled = true;
  btnText.textContent = 'Analyzing…';
  btnIcon.style.display = 'none';
  spinner.style.display = 'block';

  try {
    const res  = await fetch('/predict', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ review:text, models:activeModels, fusion:settings.fusion })
    });
    const data = await res.json();
    if (data.error) { showError(data.error); return; }

    // Add to history
    history.unshift({
      id:Date.now(), time:new Date(), review:text, result:data.result
    });
    document.getElementById('historyBadge').textContent = history.length;

    displayResult(data);
  } catch(e) {
    showError('Connection failed. Ensure Flask is running on port 5000.');
  } finally {
    btn.disabled = false;
    btnText.textContent = 'Analyze Sentiment';
    btnIcon.style.display = '';
    btnIcon.textContent   = '🔍';
    spinner.style.display = 'none';
  }
}

/* ══════════════════════════════════════════
   DISPLAY RESULT
══════════════════════════════════════════ */
const SENT_PALETTE = {
  Positive:{ col:'#00e676', glow:'rgba(0,230,118,.45)', dim:'rgba(0,230,118,.08)', bdr:'rgba(0,230,118,.28)',
             stripe:'linear-gradient(180deg,#00e676,#009944)', cls:'glow-pos' },
  Negative:{ col:'#ff4560', glow:'rgba(255,69,96,.45)',  dim:'rgba(255,69,96,.08)',  bdr:'rgba(255,69,96,.28)',
             stripe:'linear-gradient(180deg,#ff4560,#cc1133)', cls:'glow-neg' },
  Neutral: { col:'#ffbf00', glow:'rgba(255,191,0,.45)',  dim:'rgba(255,191,0,.08)',  bdr:'rgba(255,191,0,.28)',
             stripe:'linear-gradient(180deg,#ffbf00,#cc8800)', cls:'glow-neu' }
};

function displayResult(data) {
  const r = data.result, s = data.steps;
  const p = SENT_PALETTE[r.sentiment] || SENT_PALETTE.Neutral;

  /* ─ Hero ─ */
  const hc = document.getElementById('heroCard');
  hc.style.background = `linear-gradient(135deg,${p.dim},rgba(12,20,40,.97))`;
  hc.style.borderColor = p.bdr;
  hc.style.boxShadow = `0 0 50px ${p.glow.replace('.45','.12')},0 12px 50px rgba(0,0,0,.6),inset 0 1px 0 rgba(255,255,255,.05)`;
  hc.style.setProperty('--sentiment-col', p.col);

  document.getElementById('heroStripe').style.background = p.stripe;
  document.getElementById('heroHalo').style.boxShadow = `0 0 40px 14px ${p.glow}`;

  const em = document.getElementById('heroEmoji');
  em.textContent = r.emoji;
  em.style.filter = `drop-shadow(0 0 16px ${p.glow})`;

  const lbl = document.getElementById('heroLabel');
  lbl.textContent = r.detailed_label;
  lbl.style.color = p.col;
  lbl.className   = `hero-label ${p.cls}`;

  document.getElementById('heroTone').textContent = r.detailed_tone;

  document.getElementById('heroPills').innerHTML = `
    <span class="pill pill-def">${r.rec_emoji} ${r.recommendation}</span>
    <span class="pill pill-q">${r.review_quality}</span>
    <span class="pill pill-c">⚡ ${r.confidence}%</span>
    <span class="pill pill-def" style="font-size:11px;color:var(--t3)">${r.fusion_method}</span>
  `;

  /* Confidence bar */
  const conf = r.confidence;
  const cGrad = conf>=75 ? 'linear-gradient(90deg,#009944,#00e676,#6effa0)' :
                conf>=50 ? 'linear-gradient(90deg,#cc8800,#ffbf00,#ffd966)' :
                           'linear-gradient(90deg,#cc1133,#ff4560,#ff9ab0)';
  const cCol  = conf>=75 ? '#00e676' : conf>=50 ? '#ffbf00' : '#ff4560';
  const cFill = document.getElementById('confFill');
  cFill.style.background = cGrad;
  cFill.style.boxShadow  = `0 0 10px ${cCol}88`;
  document.getElementById('confNum').style.color = cCol;
  animCount('confNum', 0, conf, 900, v => `${v}%`);
  setTimeout(() => { cFill.style.width = `${conf}%`; }, 80);

  /* ─ Ring chart ─ */
  const probs = r.class_probabilities;
  const C = 2 * Math.PI * 58;  // r=58, from SVG
  const GAP = 4;

  const negA = (probs.Negative/100)*C;
  const neuA = (probs.Neutral/100)*C;
  const posA = (probs.Positive/100)*C;

  const negD = Math.max(negA-GAP, 0);
  const neuD = Math.max(neuA-GAP, 0);
  const posD = Math.max(posA-GAP, 0);

  // Set offsets immediately (no animation)
  const rNeg = document.getElementById('ringNeg');
  const rNeu = document.getElementById('ringNeu');
  const rPos = document.getElementById('ringPos');

  rNeg.setAttribute('stroke-dashoffset', '0');
  rNeu.setAttribute('stroke-dashoffset', `${-negA}`);
  rPos.setAttribute('stroke-dashoffset', `${-(negA+neuA)}`);

  // SVG rotate to start from top
  rNeg.closest('svg').style.transform = 'rotate(-90deg)';

  setTimeout(() => {
    rNeg.setAttribute('stroke-dasharray', `${negD} 9999`);
    rNeu.setAttribute('stroke-dasharray', `${neuD} 9999`);
    rPos.setAttribute('stroke-dasharray', `${posD} 9999`);
  }, 120);

  // Ring center
  const dom = Object.entries(probs).sort((a,b)=>b[1]-a[1])[0];
  const domCol = {Positive:'#00e676',Neutral:'#ffbf00',Negative:'#ff4560'};
  document.getElementById('ringNum').style.color = domCol[dom[0]]||'#eef5ff';
  document.getElementById('ringLbl').textContent  = dom[0].slice(0,3).toUpperCase();
  animCount('ringNum', 0, dom[1], 900, v=>`${v}%`);

  document.getElementById('lgPos').textContent = `${probs.Positive}%`;
  document.getElementById('lgNeu').textContent = `${probs.Neutral}%`;
  document.getElementById('lgNeg').textContent = `${probs.Negative}%`;

  /* ─ Tab 1: Preprocessing ─ */
  document.getElementById('stepOriginal').textContent = s.preprocessing.original;
  document.getElementById('stepCleaned').textContent  = s.preprocessing.cleaned || '(empty after cleaning)';
  document.getElementById('metaBefore').textContent   = s.preprocessing.word_count;
  document.getElementById('metaAfter').textContent    = s.preprocessing.clean_word_count;

  /* ─ Tab 2: VADER ─ */
  const vK = ['pos','neu','neg','compound'];
  const vL = {pos:'Positive',neu:'Neutral',neg:'Negative',compound:'Compound'};
  const vC = {pos:'#00e676',neu:'#ffbf00',neg:'#ff4560',compound:'#00cfff'};
  document.getElementById('vaderGrid').innerHTML = vK.map(k => {
    const v   = s.vader[k]!==undefined ? s.vader[k] : 0;
    const pct = k==='compound' ? Math.abs(v)*100 : v*100;
    return `<div class="vader-block">
      <div class="vader-lbl">${vL[k]}</div>
      <div class="vader-val" style="color:${vC[k]};text-shadow:0 0 14px ${vC[k]}66">${v.toFixed(4)}</div>
      <div class="vader-track"><div class="vader-fill" style="background:${vC[k]};width:${Math.max(pct,2)}%;box-shadow:0 0 6px ${vC[k]}66"></div></div>
    </div>`;
  }).join('');

  /* ─ Tab 3: Base Models ─ */
  const mMeta = {
    'Linear SVM':    {icon:'📐',desc:'TF-IDF → hyperplane distance',cls:'mc-cyan'},
    'Random Forest': {icon:'🌳',desc:'TF-IDF → 100-tree ensemble',  cls:'mc-violet'},
    'BERT':          {icon:'🧠',desc:'Dense embeddings → semantics', cls:'mc-green'}
  };
  const sC = {Positive:'#00e676',Negative:'#ff4560',Neutral:'#ffbf00'};
  document.getElementById('modelsGrid').innerHTML = Object.entries(s.models).map(([name,info]) => {
    const m = mMeta[name]||{icon:'🤖',desc:'',cls:''};
    const scores = Object.entries(info.scores).map(([k,v])=>
      `<div class="ms-row"><span class="ms-key">${k.slice(0,3)}</span><span class="ms-val" style="color:${sC[k]||'#9ab8d8'}">${v>=0?'+':''}${v.toFixed(4)}</span></div>`
    ).join('');
    return `<div class="model-card ${m.cls}">
      <div class="model-icon">${m.icon}</div>
      <div class="model-name">${name}</div>
      <div class="model-desc">${m.desc}</div>
      <div class="model-pred" style="color:${sC[info.prediction]||'#eef5ff'};text-shadow:0 0 14px ${sC[info.prediction]||'#eef5ff'}44">${info.prediction}</div>
      <div class="model-scores">${scores}</div>
    </div>`;
  }).join('');

  /* ─ Tab 4: Stacking ─ */
  const sp  = s.stacking.final_probabilities||{};
  const spN = s.stacking.meta_input_length||'?';
  const fm  = s.stacking.fusion_method||'';
  const pillMap = {Positive:'sp-pos',Neutral:'sp-neu',Negative:'sp-neg'};
  const pills = Object.entries(sp).map(([k,v])=>`<div class="sp ${pillMap[k]||''}">${k}: ${(v*100).toFixed(1)}%</div>`).join('');

  const isStacking = s.stacking.can_stack;
  document.getElementById('stackingViz').innerHTML = isStacking ? `
    <div class="stack-desc">
      The <strong>Meta-Learner</strong> (Logistic Regression) receives all <strong>${spN} numerical outputs</strong> from Step 3
      and learned which models to trust for which patterns. It multiplies those by its trained <em>weights</em> and applies Softmax.
    </div>
    <div class="stack-formula">
<span class="f-c">// Simplified Mathematical Representation</span>
<span class="f-v">Z_pos</span> = (<span class="f-v">w₁</span>·SVM) + (<span class="f-v">w₂</span>·RF) + (<span class="f-v">w₃</span>·BERT) + <span class="f-v">bias</span>
<span class="f-f">Final%</span> = <span class="f-f">Softmax</span>(<span class="f-v">Z_pos</span>, <span class="f-v">Z_neu</span>, <span class="f-v">Z_neg</span>)</div>
    <div class="stack-arrow">⬇</div>
    <div class="stack-output"><div class="stack-output-lbl">Final Meta-Learner Output</div><div class="stack-pills">${pills}</div></div>
  ` : `
    <div class="stack-desc">
      Fusion mode: <strong>${fm}</strong>. The selected models' probability outputs are combined using this method to produce the final prediction.
    </div>
    <div class="stack-arrow">⬇</div>
    <div class="stack-output"><div class="stack-output-lbl">Final Combined Output</div><div class="stack-pills">${pills}</div></div>
  `;

  /* ─ Word Map ─ */
  if (s.word_sentiments && s.word_sentiments.length) {
    document.getElementById('wordMap').innerHTML = renderWordMap(s.word_sentiments);
  }

  /* ─ Confidence warning ─ */
  document.getElementById('confWarning')?.classList.toggle('visible', conf < 55);

  /* ─ Copy button ─ */
  const copyBtnEl = document.getElementById('copyBtn');
  if (copyBtnEl) copyBtnEl.style.display = 'flex';

  /* ─ Sidebar stats ─ */
  updateSidebarStats();

  /* ─ Show & scroll ─ */
  const wr = document.getElementById('resultsWrapper');
  wr.style.display = 'block';
  switchTab(0);
  setTimeout(() => wr.scrollIntoView({behavior:'smooth',block:'nearest'}), 80);
}

/* ══════════════════════════════════════════
   TABS
══════════════════════════════════════════ */
function switchTab(i) {
  for(let j=0;j<4;j++){
    document.getElementById(`tabBtn${j}`).classList.toggle('active',j===i);
    document.getElementById(`panel${j}`).classList.toggle('active',j===i);
  }
}

/* ══════════════════════════════════════════
   HISTORY
══════════════════════════════════════════ */
function renderHistory(filter='') {
  const list = document.getElementById('histList');
  const filt = filter.toLowerCase();
  const items = history.filter(h => !filt || h.review.toLowerCase().includes(filt));

  // Always update analytics based on full history (not filtered)
  renderHistoryAnalytics();

  if (!items.length) {
    list.innerHTML = `<div class="empty-state">
      <div class="empty-icon">${filter ? '🔍' : '📋'}</div>
      <div class="empty-title">${filter ? 'No matching results' : 'No analyses yet'}</div>
      <div class="empty-desc">${filter ? 'Try a different search term.' : 'Run an analysis and results will appear here.'}</div>
    </div>`;
    return;
  }

  const sC = {Positive:'#00e676',Negative:'#ff4560',Neutral:'#ffbf00'};
  list.innerHTML = items.map((h,i) => {
    const r     = h.result;
    const excerpt = h.review.length>90 ? h.review.slice(0,90)+'…' : h.review;
    const models = r.models_used ? r.models_used.map(m=>m.toUpperCase().slice(0,3)).join(' + ') : '—';
    return `<div class="hist-item">
      <div class="hist-num">${history.indexOf(h)+1}</div>
      <span style="font-size:26px;flex-shrink:0;line-height:1">${r.emoji}</span>
      <div class="hist-body">
        <div class="hist-review">${escHtml(excerpt)}</div>
        <div class="hist-meta">
          <span>${r.fusion_method||'—'}</span>
          <div class="hist-meta-dot"></div>
          <span>${models}</span>
          <div class="hist-meta-dot"></div>
          <span>${timeAgo(h.time)}</span>
        </div>
      </div>
      <div class="hist-right">
        <div class="hist-sentiment" style="color:${sC[r.sentiment]||'#9ab8d8'}">${r.detailed_label}</div>
        <div class="hist-conf">${r.confidence}% confidence</div>
      </div>
    </div>`;
  }).join('');
}

function filterHistory() {
  const q = document.getElementById('histSearch').value;
  renderHistory(q);
}

function clearHistory() {
  history.length = 0;
  document.getElementById('historyBadge').textContent = '0';
  document.getElementById('histSearch').value = '';
  updateSidebarStats();
  renderHistory();
}

function timeAgo(d) {
  const s = Math.floor((Date.now()-d)/1000);
  if(s<60)  return 'just now';
  if(s<3600) return `${Math.floor(s/60)}m ago`;
  if(s<86400) return `${Math.floor(s/3600)}h ago`;
  return `${Math.floor(s/86400)}d ago`;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ══════════════════════
   HISTORY ANALYTICS
══════════════════════ */
function renderHistoryAnalytics() {
  const ha = document.getElementById('histAnalytics');
  const exportBtn = document.getElementById('exportBtn');
  if (!history.length) {
    if (ha) ha.style.display = 'none';
    if (exportBtn) exportBtn.style.display = 'none';
    return;
  }
  const total = history.length;
  const pos = history.filter(h => h.result.sentiment==='Positive').length;
  const neu = history.filter(h => h.result.sentiment==='Neutral').length;
  const neg = history.filter(h => h.result.sentiment==='Negative').length;
  const avgConf = (history.reduce((a,h) => a+h.result.confidence,0)/total).toFixed(1);
  if (ha) {
    ha.style.display = 'grid';
    ha.innerHTML = `
      <div class="hist-stat-card">
        <div class="hist-stat-num" style="color:var(--cyan)">${total}</div>
        <div class="hist-stat-lbl">Total</div>
      </div>
      <div class="hist-stat-card">
        <div class="hist-stat-num" style="color:#00e676">${Math.round(pos/total*100)}%</div>
        <div class="hist-stat-lbl">Positive</div>
      </div>
      <div class="hist-stat-card">
        <div class="hist-stat-num" style="color:#ffbf00">${Math.round(neu/total*100)}%</div>
        <div class="hist-stat-lbl">Neutral</div>
      </div>
      <div class="hist-stat-card">
        <div class="hist-stat-num" style="color:#ff4560">${Math.round(neg/total*100)}%</div>
        <div class="hist-stat-lbl">Negative</div>
      </div>
      <div class="hist-stat-card">
        <div class="hist-stat-num" style="color:var(--t1)">${avgConf}%</div>
        <div class="hist-stat-lbl">Avg Conf</div>
      </div>
    `;
  }
  if (exportBtn) exportBtn.style.display = '';
}

/* ══════════════════════
   WORD SENTIMENT MAP
══════════════════════ */
function renderWordMap(wordSentiments) {
  return wordSentiments.map(({word, score}) => {
    let cls = 'wm-neutral';
    if      (score >=  0.4) cls = 'wm-pos';
    else if (score >=  0.05) cls = 'wm-pos-soft';
    else if (score <= -0.4) cls = 'wm-neg';
    else if (score <= -0.05) cls = 'wm-neg-soft';
    return `<span class="wm-word ${cls}" title="Score: ${score}">${escHtml(word)}</span>`;
  }).join(' ');
}

/* ══════════════════════
   SIDEBAR STATS
══════════════════════ */
function updateSidebarStats() {
  const total = history.length;
  const el = document.getElementById('sbStatTotal');
  if (el) el.textContent = total;

  const body = document.getElementById('sbStatsBody');
  if (!body) return;

  if (!total) {
    body.innerHTML = `<div class="sb-empty-state"><div class="sb-empty-icon">📊</div><div class="sb-empty-txt">Run an analysis to see session stats.</div></div>`;
    return;
  }

  const pos = history.filter(h => h.result.sentiment==='Positive').length;
  const neu = history.filter(h => h.result.sentiment==='Neutral').length;
  const neg = history.filter(h => h.result.sentiment==='Negative').length;
  const last = history[0];

  body.innerHTML = `
    <div class="sb-stats-body">
      <div class="sb-dist-track">
        <div class="sb-dist-fill-pos" style="width:${(pos/total)*100}%"></div>
        <div class="sb-dist-fill-neu" style="width:${(neu/total)*100}%"></div>
        <div class="sb-dist-fill-neg" style="width:${(neg/total)*100}%"></div>
      </div>
      <div class="sb-dist-legend">
        <span style="color:#00e676">${pos} pos</span>
        <span style="color:#ffbf00">${neu} neu</span>
        <span style="color:#ff4560">${neg} neg</span>
      </div>
      <div class="sb-last-section">
        <div class="sb-last-lbl">Last Result</div>
        <div class="sb-last-card">
          <div style="font-size:24px;margin-bottom:5px">${last.result.emoji}</div>
          <div style="font-size:13px;font-weight:700;color:${SENT_COL[last.result.sentiment]||'#9ab8d8'};margin-bottom:3px">${last.result.detailed_label}</div>
          <div style="font-size:10.5px;color:var(--t3);font-family:'JetBrains Mono',monospace">${last.result.confidence}% · ${last.result.fusion_method}</div>
        </div>
      </div>
    </div>
  `;
}

/* ══════════════════════
   COPY RESULT
══════════════════════ */
function copyResult() {
  if (!history.length) return;
  const r = history[0].result;
  const text = [
    `Sentiment: ${r.detailed_label} (${r.confidence}%)`,
    `Recommendation: ${r.recommendation}`,
    `Tone: ${r.detailed_tone}`,
    `Fusion: ${r.fusion_method}`,
    `Models: ${(r.models_used||[]).join(', ')}`,
    `Review quality: ${r.review_quality}`
  ].join('\n');
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('copyBtn');
    if (!btn) return;
    const orig = btn.innerHTML;
    btn.innerHTML = '✓ Copied!';
    btn.style.color = 'var(--pos)';
    btn.style.borderColor = 'var(--pos-b)';
    setTimeout(() => { btn.innerHTML = orig; btn.style.color = ''; btn.style.borderColor = ''; }, 2200);
  }).catch(() => alert('Copy failed. Please copy manually.'));
}

/* ══════════════════════
   EXPORT CSV
══════════════════════ */
function exportHistory() {
  if (!history.length) return;
  const header = ['Timestamp','Review','Sentiment','Detailed Label','Confidence%','Recommendation','Models','Fusion Method','Review Quality'];
  const rows = history.map(h => {
    const r = h.result;
    return [
      new Date(h.time).toISOString(),
      `"${h.review.replace(/"/g,'""')}"`,
      r.sentiment, r.detailed_label, r.confidence,
      r.recommendation,
      `"${(r.models_used||[]).join('+')}"`,
      r.fusion_method, r.review_quality
    ];
  });
  const csv = [header, ...rows].map(r => r.join(',')).join('\n');
  const a = document.createElement('a');
  a.href = 'data:text/csv;charset=utf-8,\uFEFF' + encodeURIComponent(csv);
  a.download = `sentimentai_history_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
}

/* ══════════════════════════════════════════
   COUNTER ANIMATION
══════════════════════════════════════════ */
function animCount(id, from, to, ms, fmt) {
  const el = document.getElementById(id);
  const t0 = performance.now();
  (function step(now) {
    const t = Math.min((now-t0)/ms,1);
    const e = 1-Math.pow(1-t,3);
    el.textContent = fmt(Math.round(from+(to-from)*e));
    if(t<1) requestAnimationFrame(step);
  })(performance.now());
}

/* ══════════════════════════════════════════
   ERROR
══════════════════════════════════════════ */
function showError(msg) {
  clearError();
  document.getElementById('resultsWrapper').style.display = 'none';
  const d = document.createElement('div');
  d.className='error-toast'; d.id='errToast'; d.textContent=msg;
  document.querySelector('.input-card').appendChild(d);
  setTimeout(()=>d.remove(),6000);
}
function clearError() { document.getElementById('errToast')?.remove(); }
