/* Simplified recommender UI — reads the local DB only. No scrape controls. */
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const esc = (value) =>
  String(value ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const num = (value, digits = 0) =>
  value == null || Number.isNaN(value)
    ? '—'
    : Number(value).toLocaleString(undefined, {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });

const score2 = (value) => (value == null ? '—' : Number(value).toFixed(2));

const money = (value, currency) => {
  if (value == null) return '—';
  const symbols = { USD: '$', EUR: '€', GBP: '£', INR: '₹', LKR: 'Rs ' };
  return `${symbols[currency] || ''}${num(value, 0)}${symbols[currency] ? '' : ' ' + (currency || '')}`.trim();
};

function scoreColor(value) {
  if (value == null) return null;
  const clamped = Math.max(0, Math.min(1, value));
  return `hsl(${clamped * 132} 68% ${38 + clamped * 11}%)`;
}

function phoneLabel(phone) {
  return phone.canonical_name || phone.name || phone.raw_title || 'Unknown phone';
}

function phoneDescription(phone, maxLen = 140) {
  const text = (phone.description || '').trim();
  if (!text) return '';
  if (text.length <= maxLen) return text;
  return `${text.slice(0, maxLen - 1).trim()}…`;
}

function phoneDescriptionMarkup(phone, { compact = false } = {}) {
  const full = (phone.description || '').trim();
  if (!full) return '';
  if (compact) {
    const short = phoneDescription(phone, 160);
    return `<div class="phone-desc-block compact">
      <div class="phone-desc-label">Product description</div>
      <p class="phone-desc" title="${esc(full)}">${esc(short)}</p>
    </div>`;
  }
  return `<div class="phone-desc-block">
    <div class="phone-desc-label">Product description</div>
    <p class="phone-desc full">${esc(full)}</p>
  </div>`;
}

function phoneImageSrc(phone) {
  const id = phone.id ?? phone.smartphone_id;
  if (id) return `/phones/${id}/image`;
  return phone.image_url || '';
}

function phoneImageFailed(img) {
  const wrap = img.closest('.phone-image-wrap');
  if (!wrap || wrap.classList.contains('broken')) return;
  wrap.classList.add('broken');
  img.remove();
  const initial = (wrap.dataset.initial || '?').charAt(0).toUpperCase();
  wrap.innerHTML = `<span>${esc(initial)}</span>`;
}

function phoneImageMarkup(phone, sizeClass = '') {
  const label = phoneLabel(phone);
  const initial = esc((phone.brand || label).charAt(0).toUpperCase());
  const src = phoneImageSrc(phone);
  if (src) {
    const srcJs = JSON.stringify(src);
    const labelJs = JSON.stringify(label);
    return `<div class="phone-image-wrap ${sizeClass} zoomable" data-initial="${initial}"
      role="button" tabindex="0" title="Click to view full image"
      onclick='event.stopPropagation(); openImageLightbox(${srcJs}, ${labelJs})'
      onkeydown='if(event.key==="Enter"||event.key===" "){event.preventDefault();event.stopPropagation();openImageLightbox(${srcJs}, ${labelJs});}'>
      <img src="${esc(src)}" alt="${esc(label)}" loading="lazy"
        onerror="phoneImageFailed(this)">
    </div>`;
  }
  return `<div class="phone-image-wrap ${sizeClass} placeholder"><span>${initial}</span></div>`;
}

function openImageLightbox(src, caption = '') {
  if (!src) return;
  const box = $('#lightbox');
  const img = $('#lightbox-img');
  const cap = $('#lightbox-caption');
  if (!box || !img) return;
  img.src = src;
  img.alt = caption || 'Product image';
  if (cap) cap.textContent = caption || '';
  box.hidden = false;
  box.classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeImageLightbox() {
  const box = $('#lightbox');
  const img = $('#lightbox-img');
  if (!box) return;
  box.classList.remove('open');
  box.hidden = true;
  if (img) img.removeAttribute('src');
  document.body.style.overflow = '';
}

function starRatingMarkup(rating, count) {
  if (rating == null) return '';
  const clamped = Math.max(0, Math.min(5, Number(rating)));
  const full = Math.floor(clamped);
  const partial = clamped - full >= 0.5;
  let stars = '★'.repeat(full);
  if (partial) stars += '½';
  stars = stars.padEnd(partial ? 4 : 5, '☆');
  const countText = count != null ? ` (${num(count)})` : '';
  return `<span class="stars" title="${clamped.toFixed(1)} out of 5">${stars}</span><span class="dim small">${countText}</span>`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch { /* empty */ }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

function toast(title, message = '', kind = '') {
  const node = document.createElement('div');
  node.className = `toast ${kind}`;
  node.innerHTML = `<div class="toast-title">${esc(title)}</div>${
    message ? `<div class="toast-msg">${esc(message)}</div>` : ''
  }`;
  $('#toasts').appendChild(node);
  setTimeout(() => {
    node.style.opacity = '0';
    node.style.transition = 'all .3s';
    setTimeout(() => node.remove(), 320);
  }, 4200);
}

function emptyState(title, message) {
  return `<div class="empty">
    <div class="empty-title">${esc(title)}</div>
    <div class="empty-msg">${message}</div>
  </div>`;
}

const skeletons = (count = 3) =>
  `<div class="grid" style="gap:12px">${'<div class="skeleton"></div>'.repeat(count)}</div>`;

const THEME_KEY = 'phone-recommender-theme';

/** Lucide icons — moon = enable dark mode, sun = enable light mode */
const ICON_MOON =
  '<svg class="theme-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>';
const ICON_SUN =
  '<svg class="theme-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>';

function isDarkTheme() {
  return document.documentElement.getAttribute('data-theme') === 'dark';
}

function updateThemeButton() {
  const btn = $('#btn-theme');
  if (!btn) return;
  const dark = isDarkTheme();
  btn.innerHTML = dark ? ICON_SUN : ICON_MOON;
  const label = dark ? 'Switch to light mode' : 'Switch to dark mode';
  btn.setAttribute('aria-pressed', dark ? 'true' : 'false');
  btn.setAttribute('aria-label', label);
  btn.title = label;
}

function toggleTheme() {
  const next = isDarkTheme() ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem(THEME_KEY, next);
  updateThemeButton();
}

function initTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  if (saved === 'dark' || saved === 'light') {
    document.documentElement.setAttribute('data-theme', saved);
  }
  updateThemeButton();
}

const state = {
  aspects: [],
  health: null,
  features: [],
  weights: {},
  lastRecommend: null,
  phoneRatings: {},
  flowStep: 1,
  lastSessionEval: null,
  rankingMethod: 'weighted',
};

const VIEW_META = {
  recommend: ['Recommend', '1 Set priorities → 2 Rank → 3 Rate phones → 4 Open Evaluation tab'],
  phones: ['Phones', 'Browse smartphones with aspect scores from Amazon reviews'],
  evidence: [
    'Evidence',
    'Pipeline proof: raw review → preprocess → sentences → ABSA → scores → ranking',
  ],
  evaluation: [
    'Evaluation',
    'Simple summary of how happy people were with the phone suggestions',
  ],
};

function friendlyVerdict(verdict) {
  if (!verdict) {
    return 'No ratings yet. Go to Recommend, get a phone list, rate each phone 1–5, then come back here.';
  }
  const text = String(verdict);
  if (text.startsWith('Good')) {
    return 'Overall: Good — most people were happy with the suggested phones.';
  }
  if (text.startsWith('Mixed')) {
    return 'Overall: Okay — some people liked the suggestions, some did not. There is room to improve.';
  }
  if (text.startsWith('Weak')) {
    return 'Overall: Weak — many people were not happy with the suggested phones.';
  }
  return text;
}

function friendlyMethodLabel(method) {
  const key = String(method || '');
  if (key === 'lexicon') return 'Review text reader (rules)';
  if (key === 'lexicon_fallback') return 'Review text reader (backup)';
  if (key === 'llm') return 'AI review reader';
  return key;
}

function setFlowStep(step) {
  state.flowStep = step;
  $$('#flow-steps .flow-step').forEach((node) => {
    const n = Number(node.dataset.step);
    node.classList.toggle('active', n === step);
    node.classList.toggle('done', n < step);
  });
}

function scrollToId(id) {
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function sessionId() {
  const key = 'phone-recommender-session';
  let id = localStorage.getItem(key);
  if (!id) {
    id = `s_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
    localStorage.setItem(key, id);
  }
  return id;
}

const PRESETS = {
  Balanced: {},
  Photography: { camera: 10, display: 6, performance: 4, battery: 4, design: 5 },
  'Battery life': { battery: 10, performance: 4, price: 4, design: 3 },
  Gaming: { performance: 10, display: 8, battery: 6, design: 3 },
  'Best value': { price: 10, battery: 5, performance: 5, design: 4 },
};

function setBadges(stats) {
  $('#badge-phones').textContent = num(stats.phones);
}

async function loadHealth() {
  try {
    const health = await api('/health');
    state.health = health;
    $('#health-dot').className = 'dot ok';
    $('#health-text').textContent = `v${health.version} · DB ready`;
    $('#engine-chip').textContent = `engine: ${health.absa_engine}`;
  } catch {
    $('#health-dot').className = 'dot off';
    $('#health-text').textContent = 'API unreachable';
  }
}

async function loadBadges() {
  try {
    setBadges(await api('/stats'));
  } catch { /* ignore */ }
}

function renderWeightControls() {
  // One slider per ABSA feature (incl. design + review-based price). No separate
  // numeric "affordability" slider — budget filters cover list price.
  const keys = state.aspects.map((a) => a.aspect);
  delete state.weights.affordability;

  $('#presets').innerHTML = Object.keys(PRESETS)
    .map((name) => `<div class="preset" data-preset="${esc(name)}">${esc(name)}</div>`)
    .join('');

  $('#weights').innerHTML = keys
    .map((key) => {
      const meta = state.aspects.find((a) => a.aspect === key);
      const label = meta?.label || key;
      state.weights[key] = state.weights[key] ?? 5;
      return `<div class="weight">
        <div class="weight-head">
          <span class="weight-name">${esc(label)}</span>
          <span class="weight-mark" id="raw-${key}">${state.weights[key]} / 10</span>
        </div>
        <input type="range" min="0" max="10" step="1" value="${state.weights[key]}" data-weight="${esc(key)}">
      </div>`;
    })
    .join('');

  $$('#weights input[type="range"]').forEach((slider) => {
    slider.oninput = () => {
      state.weights[slider.dataset.weight] = Number(slider.value);
      $$('.preset').forEach((p) => p.classList.remove('active'));
      updateWeightLabels();
    };
  });

  $$('.preset').forEach((button) => {
    button.onclick = () => {
      const preset = PRESETS[button.dataset.preset];
      const isBalanced = Object.keys(preset).length === 0;
      keys.forEach((key) => {
        state.weights[key] = isBalanced ? 5 : (preset[key] ?? 0);
      });
      $$('#weights input[type="range"]').forEach((slider) => {
        slider.value = state.weights[slider.dataset.weight];
      });
      $$('.preset').forEach((p) => p.classList.toggle('active', p === button));
      updateWeightLabels();
      runRecommend();
    };
  });

  updateWeightLabels();
}

function updateWeightLabels() {
  Object.entries(state.weights).forEach(([key, value]) => {
    const markNode = $(`#raw-${key}`);
    if (markNode) markNode.textContent = `${value} / 10`;
  });
}

async function runRecommend() {
  const container = $('#rec-results');
  const evalPanel = $('#rec-eval');
  if (evalPanel) {
    evalPanel.hidden = true;
    evalPanel.innerHTML = '';
  }
  setFlowStep(2);
  container.innerHTML = skeletons(3);

  const aspectKeys = new Set(state.aspects.map((a) => a.aspect));
  const weights = Object.fromEntries(
    Object.entries(state.weights).filter(([key, value]) => aspectKeys.has(key) && value > 0)
  );

  const payload = {
    weights,
    top_k: 10,
    min_reviews: 0,
    apply_shrinkage: true,
    method: state.rankingMethod || 'weighted',
  };
  const budgetMin = $('#budget-min').value;
  const budgetMax = $('#budget-max').value;
  if (budgetMin !== '') payload.budget_min = Number(budgetMin);
  if (budgetMax !== '') payload.budget_max = Number(budgetMax);

  try {
    const response = await api('/recommend', { method: 'POST', body: JSON.stringify(payload) });
    if (!response.results.length) {
      let hint =
        'No phone passed your filters. Clear the price fields and try again.';
      try {
        const stats = await api('/stats');
        if (!stats.phones) {
          hint =
            'The database has no phones yet. Run offline ingest first, then try again.';
        } else if (!stats.aspect_sentiments) {
          hint =
            `There are ${stats.phones} phone(s) but no aspect scores yet. ` +
            'Run <code class="mono">python run.py analyze</code>, then click Rank again.';
        } else if (response.candidates_considered === 0) {
          hint =
            `Database has ${stats.phones} phone(s), but none match your budget.` +
            ' Clear the price filters and try again.';
        }
      } catch { /* keep default hint */ }
      setFlowStep(1);
      container.innerHTML = emptyState('No phones to rank', hint);
      return;
    }

    state.lastRecommend = response;
    state.phoneRatings = {};
    state.lastSessionEval = null;
    state.rankingMethod = response.method || state.rankingMethod || 'weighted';
    setFlowStep(3);
    const methodLabel =
      state.rankingMethod === 'star_rating'
        ? 'Amazon stars (baseline)'
        : 'Your priorities (proposed)';
    container.innerHTML = `
      <div class="card flow-banner">
        <div class="flow-banner-title">Step 2 — Ranked phones</div>
        <div class="row wrap small">
          <span class="dim">Showing ${response.results.length} of ${response.candidates_considered} candidates</span>
          <span class="chip info">${esc(methodLabel)}</span>
          ${Object.entries(response.weights_used || {})
            .sort((a, b) => b[1] - a[1])
            .map(([key, value]) => `<span class="chip info">${esc(key)} ${(value * 100).toFixed(0)}%</span>`)
            .join('')}
        </div>
        <p class="hint" style="margin:10px 0 0">Each phone includes a plain-language “why recommended” note. Next: rate every phone (1–5).</p>
      </div>
      ${response.results.map(renderRecommendation).join('')}
      ${renderFeedbackForm(response)}`;
    bindFeedbackForm(response);
    setTimeout(() => scrollToId('feedback-card'), 120);
  } catch (error) {
    container.innerHTML = '';
    state.lastRecommend = null;
    setFlowStep(1);
    toast('Recommendation failed', error.message, 'err');
  }
}

function renderFeedbackForm(response) {
  const rows = response.results
    .map((item) => {
      const id = item.smartphone_id;
      const stars = [1, 2, 3, 4, 5]
        .map(
          (n) =>
            `<button type="button" class="fb-star" data-phone="${id}" data-sat="${n}" aria-label="Rate ${esc(item.name)} ${n}">${n}</button>`
        )
        .join('');
      return `<div class="phone-rate-row" data-phone-row="${id}">
        <div class="phone-rate-meta">
          <span class="rank-mini">#${item.rank}</span>
          <span class="phone-rate-name">${esc(item.name)}</span>
        </div>
        <div class="feedback-stars phone-stars">${stars}</div>
      </div>`;
    })
    .join('');

  return `<div class="card pad-lg feedback-card" id="feedback-card">
    <div class="card-head"><h3>Step 3 — Rate each recommended phone</h3></div>
    <p class="hint" style="margin:0">Tap 1 (poor fit) to 5 (great fit) for every phone. These scores are saved for Evaluation and also used to check ranking quality (NDCG@3).</p>
    <div id="phone-rating-list" class="phone-rating-list">${rows}</div>
    <label class="field mt-16" style="margin-bottom:0">
      <span class="lbl">Optional comment</span>
      <textarea id="feedback-comment" rows="2" placeholder="What was missing or useful?" maxlength="2000"></textarea>
    </label>
    <div class="feedback-actions">
      <button class="btn btn-primary" id="btn-feedback" type="button">Submit scores &amp; open Evaluation →</button>
      <span class="dim small" id="feedback-status"></span>
    </div>
  </div>`;
}

function bindFeedbackForm(response) {
  $$('#phone-rating-list .fb-star').forEach((btn) => {
    btn.onclick = () => {
      const phoneId = Number(btn.dataset.phone);
      const sat = Number(btn.dataset.sat);
      state.phoneRatings[phoneId] = sat;
      $$(`#phone-rating-list .fb-star[data-phone="${phoneId}"]`).forEach((b) => {
        b.classList.toggle('active', Number(b.dataset.sat) === sat);
      });
      const rated = Object.keys(state.phoneRatings).length;
      const total = response.results.length;
      const status = $('#feedback-status');
      if (status && rated < total) status.textContent = `${rated}/${total} phones rated`;
      else if (status) status.textContent = 'All rated — ready to submit';
    };
  });
  const submit = $('#btn-feedback');
  if (!submit) return;
  submit.onclick = () => submitFeedback(response);
}

function sessionEvalHtml(phoneRatings, report) {
  const mean =
    phoneRatings.reduce((sum, r) => sum + r.satisfaction, 0) / phoneRatings.length;
  const high = phoneRatings.filter((r) => r.satisfaction >= 4).length;
  const rows = phoneRatings
    .map(
      (p) => `<tr>
        <td>#${esc(p.rank)}</td>
        <td>${esc(p.name)}</td>
        <td><strong>${esc(p.satisfaction)}</strong>/5</td>
      </tr>`
    )
    .join('');
  const verdict =
    mean >= 4
      ? 'This ranking looked like a good fit.'
      : mean >= 3
        ? 'This ranking was mixed - try adjusting priorities.'
        : 'This ranking was a weak fit - try different weights.';

  return `
    <div class="card pad-lg flow-banner ok" id="session-eval-card">
      <div class="flow-banner-title">Your scores for this ranking</div>
      <p class="hint" style="margin:0 0 12px">${esc(verdict)} These ratings were saved and will show on the Evaluation page.</p>
      <div class="eval-grid">
        <div>
          <div class="eval-stat">${score2(mean)}</div>
          <div class="eval-stat-label">Mean for this Top ${phoneRatings.length}</div>
        </div>
        <div>
          <div class="eval-stat">${high}/${phoneRatings.length}</div>
          <div class="eval-stat-label">Phones scored 4 or 5</div>
        </div>
        <div>
          <div class="eval-stat">${num(report.phone_rating_count)}</div>
          <div class="eval-stat-label">Total phone scores stored so far</div>
        </div>
      </div>
      <div class="table-wrap mt-16">
        <table>
          <thead><tr><th>Rank</th><th>Phone</th><th>Your score</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <div class="feedback-actions">
        <button class="btn btn-primary" type="button" id="btn-open-evaluation">See overall results on Evaluation →</button>
        <span class="dim small">That page shows how happy people were with all suggestions</span>
      </div>
    </div>`;
}

function bindInlineEvalActions() {
  const openBtn = $('#btn-open-evaluation');
  if (openBtn) openBtn.onclick = () => go('evaluation');
}

function renderInlineEvaluation(phoneRatings, report) {
  const evalPanel = $('#rec-eval');
  if (!evalPanel) return;
  evalPanel.hidden = false;
  // Session-only on Recommend; full cumulative model eval is on the Evaluation tab.
  evalPanel.innerHTML = sessionEvalHtml(phoneRatings, report);
  bindInlineEvalActions();
}

async function submitFeedback(response) {
  const last = response || state.lastRecommend;
  if (!last || !last.results?.length) {
    toast('Nothing to rate', 'Run Rank phones first.', 'warn');
    return;
  }
  const missing = last.results.filter((r) => !state.phoneRatings[r.smartphone_id]);
  if (missing.length) {
    toast('Rate all phones', `Still missing ${missing.length} phone rating(s).`, 'warn');
    return;
  }

  const phone_ratings = last.results.map((r) => ({
    smartphone_id: r.smartphone_id,
    name: r.name || phoneLabel(r),
    rank: r.rank,
    satisfaction: Number(state.phoneRatings[r.smartphone_id]),
    final_score: r.final_score ?? null,
  }));
  const mean =
    phone_ratings.reduce((sum, r) => sum + r.satisfaction, 0) / phone_ratings.length;

  const status = $('#feedback-status');
  const btn = $('#btn-feedback');
  if (btn) btn.disabled = true;
  if (status) status.textContent = 'Saving scores…';
  try {
    await api('/feedback', {
      method: 'POST',
      body: JSON.stringify({
        satisfaction: Math.max(1, Math.min(5, Math.round(mean))),
        comment: ($('#feedback-comment')?.value || '').trim() || null,
        weights_used: last.weights_used || {},
        top_phone_ids: last.results.map((r) => r.smartphone_id).filter(Boolean),
        top_phone_names: last.results.map((r) => r.name || phoneLabel(r)),
        phone_ratings,
        candidates_considered: last.candidates_considered ?? null,
        session_id: sessionId(),
        ranking_method: last.method || state.rankingMethod || 'weighted',
      }),
    });

    const report = await api('/feedback/evaluation');
    state.lastSessionEval = { phone_ratings, report };
    setFlowStep(4);
    renderInlineEvaluation(phone_ratings, report);

    if (status) status.textContent = 'Saved. Opening Evaluation tab…';
    if (btn) btn.textContent = 'Scores submitted';
    toast(
      'Added to Evaluation',
      `${report.phone_rating_count} phone score(s) across ${report.feedback_count} session(s).`,
      'ok'
    );
    // Full model evaluation lives on its own tab.
    go('evaluation');
  } catch (error) {
    if (btn) btn.disabled = false;
    if (status) status.textContent = '';
    toast('Feedback failed', error.message, 'err');
  }
}

function evaluationMarkup(report, { compact = false } = {}) {
  const dist = report.satisfaction_distribution || {};
  const distMax = Math.max(1, ...Object.values(dist).map(Number));
  const methods = Object.entries(report.absa_method_breakdown || {})
    .map(([m, c]) => `<span class="chip info">${esc(friendlyMethodLabel(m))}: ${num(c)}</span>`)
    .join('') || '<span class="dim">No review analysis yet</span>';
  const absa = report.absa_validation || {};
  const byRating = Object.entries(absa.by_rating || {})
    .map(
      ([stars, row]) =>
        `<tr>
          <td>${esc(stars)}★ reviews</td>
          <td>${num(row.reviews)}</td>
          <td>${(row.agreement_rate * 100).toFixed(1)}% match</td>
          <td>${score2(row.mean_absolute_error)}</td>
        </tr>`
    )
    .join('');
  const byRank = Object.entries(report.mean_satisfaction_by_rank || {})
    .map(
      ([rank, mean]) =>
        `<tr><td>Phone placed #${esc(rank)}</td><td>${score2(mean)} out of 5</td></tr>`
    )
    .join('');
  const recent = (report.recent_feedback || [])
    .map((f) => {
      const when = f.created_at ? new Date(f.created_at).toLocaleString() : '—';
      const phones =
        Array.isArray(f.phone_ratings) && f.phone_ratings.length
          ? f.phone_ratings
              .map(
                (p) =>
                  `#${esc(p.rank)} ${esc(p.name)} — rated <strong>${esc(p.satisfaction)}</strong> out of 5`
              )
              .join('<br>')
          : esc(f.comment || 'No per-phone ratings');
      return `<div class="feedback-item">
        <div class="meta">${esc(when)} · average for that list: <strong>${esc(f.satisfaction)}</strong>/5 · ${num((f.phone_ratings || []).length)} phone(s) rated</div>
        <div>${phones}</div>
        ${f.comment ? `<div class="dim small" style="margin-top:8px">${esc(f.comment)}</div>` : ''}
      </div>`;
    })
    .join('');

  const highPct =
    report.high_satisfaction_rate == null
      ? '—'
      : `${(report.high_satisfaction_rate * 100).toFixed(0)}%`;

  const meanLabel =
    report.mean_satisfaction == null ? '—' : score2(report.mean_satisfaction);
  const top1Label =
    report.top1_mean_satisfaction == null ? '—' : score2(report.top1_mean_satisfaction);
  const ndcgLabel =
    report.mean_ndcg_at_3 == null ? '—' : score2(report.mean_ndcg_at_3);
  const spearmanLabel =
    report.mean_spearman == null ? '—' : score2(report.mean_spearman);

  return `
    <div class="card pad-lg banner info" style="margin-bottom:16px" id="final-eval-card">
      <div>
        <strong>How good are the phone suggestions?</strong>
        <div>${esc(friendlyVerdict(report.evaluation_verdict))}</div>
        <p class="hint" style="margin:10px 0 0">
          Built from <strong>${num(report.phone_rating_count)}</strong> phone ratings
          across <strong>${num(report.feedback_count)}</strong> times people used Rank.
          The more people rate, the clearer this picture becomes.
        </p>
      </div>
    </div>
    <div class="card pad-lg banner info" style="margin-bottom:16px">
      <div>
        <strong>Did the system put the best-fitting phones near the top?</strong>
        <div>${esc(report.ranking_quality_note || '')}</div>
        <p class="hint" style="margin:10px 0 0">
          Your 1–5 phone ratings are also used as “how well each phone fits my needs” to measure ranking quality (NDCG@3).
          <strong>1.00</strong> = ideal order · closer to <strong>0</strong> = poor order.
        </p>
      </div>
    </div>
    <div class="eval-grid">
      <div class="card pad-lg">
        <div class="card-head"><h3>Ranking quality (NDCG@3)</h3></div>
        <div class="eval-stat">${ndcgLabel}</div>
        <div class="eval-stat-label">Average over ${num(report.ndcg_session_count)} ranking run(s) with per-phone scores. Checks whether the top 3 system picks match phones you rated highest.</div>
        <div class="row wrap mt-16" style="gap:8px">
          <span class="chip info">Order match (Spearman): ${spearmanLabel}</span>
          <span class="chip info">From ${num(report.spearman_session_count)} run(s)</span>
        </div>
        <p class="hint mt-16" style="margin:0">Spearman: +1 similar order · 0 no clear link · −1 opposite order. Shown when enough phones were rated in a run.</p>
      </div>
      <div class="card pad-lg">
        <div class="card-head"><h3>Proposed method vs Amazon-star baseline</h3></div>
        <p class="hint" style="margin:0 0 12px">${esc(report.baseline_comparison_note || '')}</p>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Method</th><th>Rated runs</th><th>Avg phone rating</th><th>NDCG@3</th></tr></thead>
            <tbody>
              <tr>
                <td>Your priorities (proposed)</td>
                <td>${num(report.proposed_session_count)}</td>
                <td>${report.proposed_mean_satisfaction == null ? '—' : score2(report.proposed_mean_satisfaction) + '/5'}</td>
                <td>${report.proposed_mean_ndcg_at_3 == null ? '—' : score2(report.proposed_mean_ndcg_at_3)}</td>
              </tr>
              <tr>
                <td>Amazon stars (baseline)</td>
                <td>${num(report.baseline_session_count)}</td>
                <td>${report.baseline_mean_satisfaction == null ? '—' : score2(report.baseline_mean_satisfaction) + '/5'}</td>
                <td>${report.baseline_mean_ndcg_at_3 == null ? '—' : score2(report.baseline_mean_ndcg_at_3)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
      <div class="card pad-lg">
        <div class="card-head"><h3>Average happiness with suggested phones</h3></div>
        <div class="eval-stat">${meanLabel}<span class="eval-stat-unit"> / 5</span></div>
        <div class="eval-stat-label">1 = poor fit · 5 = great fit. Based on ${num(report.phone_rating_count)} ratings from ${num(report.feedback_count)} ranking runs.</div>
        <div class="row wrap mt-16" style="gap:8px">
          <span class="chip info">#1 phone average: ${top1Label}/5</span>
          <span class="chip info">Liked (4 or 5): ${highPct}</span>
        </div>
        <p class="hint mt-16" style="margin:0">How many times each score was given:</p>
        <div class="mt-16">
          ${[5, 4, 3, 2, 1]
            .map((n) => {
              const c = Number(dist[String(n)] || 0);
              const pct = (c / distMax) * 100;
              const label = n === 5 ? 'Great (5)' : n === 4 ? 'Good (4)' : n === 3 ? 'Okay (3)' : n === 2 ? 'Weak (2)' : 'Poor (1)';
              return `<div class="dist-row"><span>${label}</span><div class="dist-track"><div class="dist-fill" style="width:${pct}%"></div></div><span class="dim">${c}</span></div>`;
            })
            .join('')}
        </div>
      </div>
      <div class="card pad-lg">
        <div class="card-head"><h3>Were higher-ranked phones liked more?</h3></div>
        <p class="hint" style="margin:0 0 12px">If the system is working well, phones placed near the top (#1, #2) should usually get higher ratings than phones lower on the list.</p>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Place in the list</th><th>Average rating from users</th></tr></thead>
            <tbody>${byRank || '<tr><td colspan="2" class="dim">No ratings yet — rate phones after ranking</td></tr>'}</tbody>
          </table>
        </div>
      </div>
    </div>
    <div class="eval-grid mt-16">
      <div class="card pad-lg">
        <div class="card-head"><h3>Do review opinions match star ratings?</h3></div>
        <div class="eval-stat">${((absa.agreement_rate || 0) * 100).toFixed(1)}%<span class="eval-stat-unit"> match</span></div>
        <div class="eval-stat-label">Checked on ${num(absa.reviews_compared)} Amazon reviews. Higher % means the text analysis usually agrees with the reviewer’s star score.</div>
        <div class="row wrap mt-16" style="gap:8px">${methods}</div>
        <p class="hint mt-16" style="margin:0">Current reading mode: ${esc(report.absa_engine === 'llm' ? 'AI' : report.absa_engine)}</p>
      </div>
    </div>
    <div class="card pad-lg mt-16">
      <div class="card-head"><h3>Match quality for 1★ to 5★ reviews</h3></div>
      <p class="hint" style="margin:0 0 12px">
        For each star level: how often the text analysis agreed with the stars, and how far off it was on average (smaller “difference” is better).
      </p>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Review stars</th><th>How many reviews</th><th>How often they matched</th><th>Average difference</th></tr></thead>
          <tbody>${byRating || '<tr><td colspan="4" class="dim">No review checks yet</td></tr>'}</tbody>
        </table>
      </div>
    </div>
    <div class="card pad-lg mt-16">
      <div class="card-head"><h3>Latest ratings people gave</h3></div>
      <div class="feedback-list">${recent || '<p class="dim">No ratings yet. Use Recommend, then score each suggested phone.</p>'}</div>
    </div>`;
}

function chartsMarkup(chartPayload) {
  const charts = chartPayload?.charts || [];
  const note = chartPayload?.note || '';
  if (!charts.length) {
    return `<div class="card pad-lg mt-16">
      <div class="card-head"><h3>Research charts</h3></div>
      <p class="hint" style="margin:0">${esc(note || 'No charts available yet.')}</p>
    </div>`;
  }
  const cards = charts
    .map(
      (c) => `<figure class="eval-chart-card" data-chart-url="${esc(c.url)}" data-chart-title="${esc(c.title)}">
        <div class="eval-chart-frame">
          <img src="${esc(c.url)}?v=${encodeURIComponent(state.health?.version || '1')}" alt="${esc(c.title)}" loading="lazy">
        </div>
        <figcaption>
          <strong>${esc(c.title)}</strong>
          <span class="hint">${esc(c.caption || '')}</span>
        </figcaption>
      </figure>`
    )
    .join('');
  return `<div class="card pad-lg mt-16" id="eval-charts">
      <div class="card-head">
        <h3>Explainable research charts</h3>
        <span class="chip info">${num(charts.length)} figures</span>
      </div>
      <p class="hint" style="margin:0 0 14px">${esc(note)} Click any chart to enlarge.</p>
      <div class="eval-chart-grid">${cards}</div>
    </div>`;
}

function bindEvaluationCharts() {
  $$('.eval-chart-card').forEach((card) => {
    card.onclick = () => {
      const url = card.dataset.chartUrl;
      const title = card.dataset.chartTitle || 'Research chart';
      if (!url) return;
      openImageLightbox(url, title);
    };
  });
}

function brandBarsMarkup(brands) {
  const max = Math.max(...(brands || []).map((b) => b.phones), 1);
  const rows = (brands || [])
    .map(
      (b) => `<div class="ev-brand-row">
        <span>${esc(b.brand)}</span>
        <div class="ev-brand-track"><div class="ev-brand-fill" style="width:${(b.phones / max) * 100}%"></div></div>
        <span class="mono">${num(b.phones)}</span>
      </div>`
    )
    .join('');
  return rows || '<p class="dim">No brand data.</p>';
}

function funnelMarkup(funnel) {
  const items = [
    ['Phones', funnel.phones],
    ['Raw reviews', funnel.raw_reviews],
    ['Kept', funnel.kept_reviews],
    ['Sentences', funnel.sentences],
    ['Aspect labels', funnel.aspect_labels],
    ['Scored phones', funnel.scored_phones],
  ];
  return `<div class="ev-funnel">${items
    .map(
      ([label, value]) => `<div class="ev-funnel-item">
        <div class="n">${num(value)}</div>
        <div class="l">${esc(label)}</div>
      </div>`
    )
    .join('')}</div>`;
}

function tableFromRows(rows) {
  if (!rows?.length) return '<p class="dim">No sample rows for this stage.</p>';
  const keys = Object.keys(rows[0]);
  const head = keys.map((k) => `<th>${esc(k)}</th>`).join('');
  const body = rows
    .map(
      (row) =>
        `<tr>${keys
          .map((k) => {
            const v = row[k];
            const text = v == null ? '' : String(v);
            return `<td>${esc(text.length > 160 ? `${text.slice(0, 160)}…` : text)}</td>`;
          })
          .join('')}</tr>`
    )
    .join('');
  return `<div class="ev-table-wrap"><table class="ev-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function walkthroughMarkup(wt) {
  if (!wt) return '';
  const sent = (wt.sentences || [])
    .map((s) => `<li><span class="mono dim">#${s.sentence_id}</span> ${esc(s.text)}</li>`)
    .join('');
  const aspects = (wt.aspect_sentiments || [])
    .map(
      (a) =>
        `<span class="chip info">${esc(a.aspect)} · ${esc(a.sentiment)}${
          a.opinion_term ? ` · ${esc(a.opinion_term)}` : ''
        }</span>`
    )
    .join(' ');
  const scores = (wt.feature_scores || [])
    .map(
      (s) =>
        `<div class="bar-row" style="grid-template-columns:100px 1fr 72px">
          <span class="bar-name">${esc(s.aspect)}</span>
          <span class="bar-track"><span class="bar-fill" style="width:${(s.score || 0) * 100}%;background:${scoreColor(s.score)}"></span></span>
          <span class="bar-val">${score2(s.score)} <span class="dim">n=${num(s.mentions)}</span></span>
        </div>`
    )
    .join('');
  const contrib = (wt.user_weighted_contribution || [])
    .map(
      (c) =>
        `<tr><td>${esc(c.aspect)}</td><td>${score2(c.score)}</td><td>${score2(c.weight)}</td><td>${score2(c.contribution)}</td></tr>`
    )
    .join('');
  return `<div class="card pad-lg mt-16 ev-walk" id="ev-walkthrough">
    <div class="card-head"><h3>One real review through the full pipeline</h3></div>
    <p class="hint" style="margin:0 0 8px">Use this in a viva: walk top → bottom to prove each stage ran on real data.
      Phone: <strong>${esc(wt.phone?.name || '—')}</strong> (${esc(wt.phone?.brand || '—')}).</p>
    <div class="ev-block">
      <h4>1 · Raw Amazon review</h4>
      <p class="small dim" style="margin:0 0 4px">Review #${num(wt.raw_review?.review_id)} · rating ${esc(String(wt.raw_review?.rating ?? '—'))} · ${esc(wt.raw_review?.language || '')}</p>
      <p style="margin:0"><strong>${esc(wt.raw_review?.title || '')}</strong><br>${esc(wt.raw_review?.body || '')}</p>
    </div>
    <div class="ev-block">
      <h4>2 · Preprocessing decision</h4>
      <p style="margin:0"><span class="chip info">${esc(wt.preprocess?.decision || '')}</span>
        ${wt.preprocess?.excluded_reason ? ` reason: ${esc(wt.preprocess.excluded_reason)}` : ' → kept for analysis'}</p>
      <p class="small" style="margin:8px 0 0">${esc(wt.preprocess?.cleaned_body || '')}</p>
    </div>
    <div class="ev-block">
      <h4>3 · Segmented sentences</h4>
      <ul style="margin:0;padding-left:18px">${sent || '<li class="dim">None</li>'}</ul>
    </div>
    <div class="ev-block">
      <h4>4 · Detected aspect → sentiment</h4>
      <div class="row wrap" style="gap:6px">${aspects || '<span class="dim">None</span>'}</div>
    </div>
    <div class="ev-block">
      <h4>5 · Feature scores (this phone)</h4>
      ${scores || '<p class="dim">No scores</p>'}
    </div>
    <div class="ev-block">
      <h4>6 · User-weighted contribution → ranking input</h4>
      <div class="ev-table-wrap"><table class="ev-table">
        <thead><tr><th>Aspect</th><th>Score</th><th>Weight</th><th>Contribution</th></tr></thead>
        <tbody>${contrib}</tbody>
      </table></div>
      <p class="hint" style="margin:8px 0 0">${esc(wt.explanation_hint || '')}</p>
    </div>
  </div>`;
}

function evidenceMarkup(summary, stagePayload, walkthrough) {
  const funnel = summary.funnel || {};
  const brands = summary.brands || [];
  const stages = summary.stages || [];
  const chartUrl = `${summary.brand_chart_url || '/ui/charts/01_phone_brand_distribution.png'}?v=${encodeURIComponent(state.health?.version || '1')}`;
  const activeId = stagePayload?.stage || stages[0]?.id || 'raw';
  const activeMeta = stages.find((s) => s.id === activeId) || stagePayload?.meta || {};

  const stepButtons = stages
    .map(
      (s, i) => `<button type="button" class="ev-step ${s.id === activeId ? 'active' : ''}" data-stage="${esc(s.id)}">
        <span class="ev-num">${i + 1}</span>
        <span class="ev-title">${esc(s.title)}</span>
      </button>`
    )
    .join('');

  const downloads = [...stages, ...(summary.extra_downloads || [])]
    .map((s) => {
      const file = s.file;
      const title = s.title || file;
      const href = s.download_url || `/evidence/download/${file}`;
      const ready = s.exists !== false;
      return `<div class="ev-dl-card">
        <div class="name">${esc(title)}</div>
        <div class="file">${esc(file)}</div>
        ${
          ready
            ? `<a class="btn btn-sm" href="${esc(href)}" download>Download CSV</a>`
            : `<button class="btn btn-sm" type="button" disabled>Not generated yet</button>`
        }
      </div>`;
    })
    .join('');

  return `
    <div class="card pad-lg">
      <div class="card-head">
        <h3>Corpus funnel (proof numbers)</h3>
        <button class="btn btn-sm" type="button" id="btn-ev-export">Regenerate CSVs</button>
      </div>
      <p class="hint" style="margin:0 0 12px">${esc(summary.note || '')}</p>
      ${funnelMarkup(funnel)}
      <p class="dim small" style="margin:10px 0 0">CSV folder: <code class="mono">${esc(summary.folder || '')}</code></p>
      <span class="dim small" id="ev-export-status"></span>
    </div>

    <div class="grid cols-2 mt-16" style="gap:16px;align-items:start">
      <div class="card pad-lg">
        <div class="card-head"><h3>Brand distribution</h3></div>
        <p class="hint" style="margin:0 0 8px">How many phones per manufacturer after carrier/manual cleanup.</p>
        ${brandBarsMarkup(brands)}
      </div>
      <div class="card pad-lg">
        <div class="card-head"><h3>Brand chart</h3></div>
        <figure class="eval-chart-card" data-chart-url="${esc(summary.brand_chart_url || '/ui/charts/01_phone_brand_distribution.png')}" data-chart-title="Phone distribution by brand" style="margin:0">
          <div class="eval-chart-frame">
            <img src="${esc(chartUrl)}" alt="Brand distribution" loading="lazy">
          </div>
          <figcaption><strong>Phone distribution by brand</strong><span class="hint">Click to enlarge</span></figcaption>
        </figure>
      </div>
    </div>

    <div class="card pad-lg mt-16">
      <div class="card-head"><h3>Pipeline — click each stage</h3></div>
      <p class="hint" style="margin:0 0 4px">Raw review → preprocess → sentence → aspect + sentiment → feature score → weighted contribution → ranking.</p>
      <div class="ev-pipeline" id="ev-pipeline">${stepButtons}</div>
      <div class="mt-16" id="ev-stage-panel">
        <h4 style="margin:0 0 6px">${esc(activeMeta.title || activeId)}</h4>
        <p class="hint" style="margin:0 0 8px">${esc(activeMeta.what || '')} <em>${esc(activeMeta.show || '')}</em></p>
        ${
          activeMeta.download_url
            ? `<p style="margin:0 0 8px"><a class="btn btn-sm" href="${esc(activeMeta.download_url)}" download>Download this stage CSV</a></p>`
            : ''
        }
        <div id="ev-stage-table">${tableFromRows(stagePayload?.rows || [])}</div>
      </div>
    </div>

    ${walkthroughMarkup(walkthrough)}

    <div class="card pad-lg mt-16">
      <div class="card-head"><h3>Download evidence CSVs</h3></div>
      <p class="hint" style="margin:0">If examiners ask for the preprocessed dataset, give them <strong>03_kept_reviews.csv</strong>.
        For the full audit trail (kept + dropped + reason), give <strong>02_preprocessed_reviews.csv</strong>.</p>
      <div class="ev-dl-grid">${downloads}</div>
    </div>`;
}

async function loadEvidenceStage(stageId) {
  return api(`/evidence/stage/${encodeURIComponent(stageId)}?limit=8`);
}

function bindEvidence(summary) {
  bindEvaluationCharts();
  const exportBtn = $('#btn-ev-export');
  if (exportBtn) {
    exportBtn.onclick = async () => {
      const status = $('#ev-export-status');
      exportBtn.disabled = true;
      if (status) status.textContent = 'Generating…';
      try {
        const result = await api('/evidence/export', { method: 'POST', body: '{}' });
        if (status) status.textContent = `Wrote ${result.count} files.`;
        toast('Evidence CSVs ready', `${result.count} files in evidence folder`, 'ok');
        await renderEvidence();
      } catch (error) {
        if (status) status.textContent = error.message;
        toast('Export failed', error.message, 'err');
      } finally {
        exportBtn.disabled = false;
      }
    };
  }

  $$('#ev-pipeline .ev-step').forEach((btn) => {
    btn.onclick = async () => {
      const stageId = btn.dataset.stage;
      $$('#ev-pipeline .ev-step').forEach((b) => b.classList.toggle('active', b === btn));
      const panel = $('#ev-stage-panel');
      if (panel) panel.innerHTML = '<p class="dim">Loading…</p>';
      try {
        const stagePayload = await loadEvidenceStage(stageId);
        const meta = summary.stages.find((s) => s.id === stageId) || stagePayload.meta || {};
        if (panel) {
          panel.innerHTML = `
            <h4 style="margin:0 0 6px">${esc(meta.title || stageId)}</h4>
            <p class="hint" style="margin:0 0 8px">${esc(meta.what || '')} <em>${esc(meta.show || '')}</em></p>
            ${
              meta.download_url
                ? `<p style="margin:0 0 8px"><a class="btn btn-sm" href="${esc(meta.download_url)}" download>Download this stage CSV</a></p>`
                : ''
            }
            <div id="ev-stage-table">${tableFromRows(stagePayload.rows || [])}</div>`;
        }
      } catch (error) {
        if (panel) panel.innerHTML = `<p class="dim">${esc(error.message)}</p>`;
      }
    };
  });
}

async function renderEvidence() {
  const root = $('#evidence-content');
  if (!root) return;
  root.innerHTML = skeletons(3);
  try {
    let summary = await api('/evidence/summary');
    const missing = (summary.stages || []).some((s) => s.exists === false);
    if (missing) {
      try {
        await api('/evidence/export', { method: 'POST', body: '{}' });
        summary = await api('/evidence/summary');
      } catch { /* show UI anyway */ }
    }
    const firstStage = summary.stages?.[0]?.id || 'raw';
    const [stagePayload, walkthrough] = await Promise.all([
      loadEvidenceStage(firstStage),
      api('/evidence/walkthrough').catch(() => null),
    ]);
    root.innerHTML = evidenceMarkup(summary, stagePayload, walkthrough);
    bindEvidence(summary);
  } catch (error) {
    const msg = String(error.message || error);
    const hint =
      msg === 'Not Found' || msg.includes('404')
        ? 'The Evidence API is missing — restart the server with <code class="mono">python run.py serve</code>, then hard-refresh this page (Ctrl+F5).'
        : esc(msg);
    root.innerHTML = emptyState('Evidence unavailable', hint);
  }
}

async function renderEvaluation() {
  const root = $('#eval-content');
  root.innerHTML = skeletons(3);
  try {
    const [report, charts] = await Promise.all([
      api('/feedback/evaluation'),
      api('/feedback/charts').catch(() => ({ charts: [], note: 'Charts could not be loaded.' })),
    ]);
    root.innerHTML = evaluationMarkup(report) + chartsMarkup(charts);
    bindEvaluationCharts();
  } catch (error) {
    root.innerHTML = emptyState('Evaluation unavailable', esc(error.message));
  }
}

function restoreRecommendView() {
  setFlowStep(state.flowStep || 1);
  const evalPanel = $('#rec-eval');
  if (!state.lastRecommend) {
    if (evalPanel) {
      evalPanel.hidden = true;
      evalPanel.innerHTML = '';
    }
    return;
  }
  const response = state.lastRecommend;
  const container = $('#rec-results');
  container.innerHTML = `
    <div class="card flow-banner">
      <div class="flow-banner-title">Step 2 — Ranked phones</div>
      <div class="row wrap small">
        <span class="dim">Showing ${response.results.length} of ${response.candidates_considered} candidates</span>
      </div>
    </div>
    ${response.results.map(renderRecommendation).join('')}
    ${renderFeedbackForm(response)}`;
  bindFeedbackForm(response);
  Object.entries(state.phoneRatings).forEach(([phoneId, sat]) => {
    $$(`#phone-rating-list .fb-star[data-phone="${phoneId}"]`).forEach((b) => {
      b.classList.toggle('active', Number(b.dataset.sat) === Number(sat));
    });
  });
  if (state.lastSessionEval && evalPanel) {
    renderInlineEvaluation(
      state.lastSessionEval.phone_ratings,
      state.lastSessionEval.report
    );
    setFlowStep(4);
  } else if (evalPanel) {
    evalPanel.hidden = true;
  }
}

function renderRecommendation(item) {
  const rows = item.breakdown
    .filter((c) => c.weight > 0)
    .map(
      (c) => `<div class="bar-row" style="grid-template-columns:120px 1fr 92px">
        <span class="bar-name">${esc(c.aspect)}</span>
        <span class="bar-track"><span class="bar-fill"
          style="width:${(c.score ?? 0) * 100}%;background:${scoreColor(c.score)}"></span></span>
        <span class="bar-val">${score2(c.score)} <span class="dim">×${score2(c.weight)}</span></span>
      </div>`
    )
    .join('');

  const amazonRating = starRatingMarkup(item.site_rating, item.site_rating_count);

  return `<div class="rec ${item.rank === 1 ? 'top' : ''}">
    <div class="rec-layout">
      ${phoneImageMarkup(item, 'rec-size')}
      <div class="rec-body">
        <div class="rec-head">
          <div class="rank">${item.rank}</div>
          <div style="min-width:0;flex:1">
            <div class="phone-name">${esc(item.name)}</div>
            <div class="phone-meta">${esc(item.brand || 'Unknown')}</div>
            ${amazonRating ? `<div class="phone-rating-row">${amazonRating}</div>` : ''}
            <div class="phone-meta">${money(item.price, item.currency)} · ${num(item.review_count)} analysed reviews</div>
          </div>
          <div class="rec-score">
            <div class="rec-score-val" style="color:${scoreColor(item.final_score)}">${item.final_score.toFixed(3)}</div>
            <div class="rec-score-lbl">match score</div>
          </div>
        </div>
        <div class="mt-16">${rows}</div>
        ${
          item.explanation
            ? `<div class="rec-explain mt-16"><strong>Why recommended:</strong> ${esc(item.explanation)}</div>`
            : ''
        }
        <div class="row wrap mt-16">
          ${item.strengths.map((s) => `<span class="chip pos">▲ ${esc(s)}</span>`).join('')}
          ${item.weaknesses.map((w) => `<span class="chip neg">▼ ${esc(w)}</span>`).join('')}
          <div class="spacer"></div>
          <button class="btn btn-sm btn-ghost" onclick="openPhone(${item.smartphone_id})">Details →</button>
        </div>
      </div>
    </div>
  </div>`;
}

async function renderPhones() {
  const grid = $('#phone-grid');
  grid.innerHTML = skeletons(4);
  const query = $('#phone-search').value.trim();
  const brand = $('#phone-brand').value;
  const params = new URLSearchParams({ limit: '200' });
  if (query) params.set('q', query);
  if (brand) params.set('brand', brand);

  try {
    const [list, features] = await Promise.all([
      api(`/phones?${params}`),
      state.features.length ? Promise.resolve(state.features) : api('/features'),
    ]);
    state.features = features;
    $('#phone-count').textContent = `${list.total} phone(s)`;

    if (!list.items.length) {
      grid.innerHTML = emptyState(
        'No phones in the database',
        'Run <code class="mono">python run.py ingest-hf</code> then refresh.'
      );
      return;
    }

    const byId = Object.fromEntries(features.map((f) => [f.smartphone_id, f]));
    const aspects = state.aspects.map((a) => a.aspect);

    grid.innerHTML = list.items
      .map((phone) => {
        const vector = byId[phone.id];
        const bars = aspects
          .map((aspect) => {
            const value = vector?.scores[aspect];
            return `<div class="bar-row" style="grid-template-columns:78px 1fr 38px;margin-bottom:6px">
              <span class="bar-name">${esc(aspect)}</span>
              <span class="bar-track"><span class="bar-fill"
                style="width:${(value ?? 0) * 100}%;background:${scoreColor(value) || 'var(--surface-3)'}"></span></span>
              <span class="bar-val" style="font-size:11px">${score2(value)}</span>
            </div>`;
          })
          .join('');
        return `<div class="phone-card" onclick="openPhone(${phone.id})">
          ${phoneImageMarkup(phone, 'card-size')}
          <div class="phone-card-body">
            <div class="phone-name truncate" title="${esc(phoneLabel(phone))}">${esc(phoneLabel(phone))}</div>
            <div class="phone-meta">${esc(phone.brand || 'Unknown')}</div>
            ${phone.site_rating != null ? `<div class="phone-rating-row">${starRatingMarkup(phone.site_rating, phone.site_rating_count)}</div>` : ''}
            <div class="phone-price">${money(phone.latest_price, phone.currency)}</div>
            <div class="phone-meta">${num(phone.analyzed_review_count)} analysed reviews</div>
            ${phoneDescriptionMarkup(phone, { compact: true })}
            ${bars}
          </div>
        </div>`;
      })
      .join('');
  } catch (error) {
    grid.innerHTML = '';
    toast('Could not load phones', error.message, 'err');
  }
}

async function openPhone(phoneId) {
  $('#drawer-body').innerHTML = skeletons(2);
  $('#drawer').classList.add('open');
  $('#drawer-backdrop').classList.add('open');

  try {
    const phone = await api(`/phones/${phoneId}`);
    $('#drawer-title').textContent = phoneLabel(phone);
    const ratingLine = phone.site_rating != null
      ? ` · ${phone.site_rating.toFixed(1)}★ (${num(phone.site_rating_count)})`
      : '';
    $('#drawer-sub').textContent =
      `${phone.brand || 'Unknown'} · ${money(phone.latest_price, phone.currency)} · ` +
      `${num(phone.analyzed_review_count)} analysed reviews${ratingLine}`;

    const hero = phone.image_url
      ? `<div class="drawer-hero">${phoneImageMarkup(phone, 'drawer-size')}</div>`
      : '';

    const scoreByAspect = Object.fromEntries(
      (phone.aspect_scores || []).map((s) => [s.aspect, s])
    );
    const aspectOrder = state.aspects.length
      ? state.aspects.map((a) => a.aspect)
      : Object.keys(scoreByAspect);
    const breakdown = aspectOrder.length
      ? aspectOrder
          .map((aspect) => {
            const s = scoreByAspect[aspect];
            const label = state.aspects.find((a) => a.aspect === aspect)?.label || aspect;
            if (!s || !s.mention_count) {
              return `<div style="margin-bottom:14px">
        <div class="row" style="margin-bottom:6px">
          <strong class="cap">${esc(label)}</strong>
          <span class="dim small">no review mentions yet</span>
          <div class="spacer"></div>
          <span class="mono dim">—</span>
        </div>
        <div class="stacked" style="height:16px"></div>
      </div>`;
            }
            return `<div style="margin-bottom:14px">
        <div class="row" style="margin-bottom:6px">
          <strong class="cap">${esc(label)}</strong>
          <span class="dim small">${num(s.mention_count)} mentions</span>
          <div class="spacer"></div>
          <span class="mono" style="color:${scoreColor(s.score)}"><strong>${score2(s.score)}</strong></span>
        </div>
        <div class="stacked" style="height:16px">
          ${s.positive_count ? `<div style="width:${(s.positive_count / s.mention_count) * 100}%;background:var(--positive)"></div>` : ''}
          ${s.neutral_count ? `<div style="width:${(s.neutral_count / s.mention_count) * 100}%;background:var(--neutral)"></div>` : ''}
          ${s.negative_count ? `<div style="width:${(s.negative_count / s.mention_count) * 100}%;background:var(--negative)"></div>` : ''}
        </div>
      </div>`;
          })
          .join('')
      : '<p class="dim small">No aspect scores yet — run analyze.</p>';

    $('#drawer-body').innerHTML = `${hero}
      ${phoneDescriptionMarkup(phone)}
      <div class="mt-16">${breakdown}</div>
      <p class="card-note mt-16">Scores come from review sentences already stored in the database.</p>`;
  } catch (error) {
    $('#drawer-body').innerHTML = `<div class="banner">${esc(error.message)}</div>`;
  }
}

function closeDrawer() {
  $('#drawer').classList.remove('open');
  $('#drawer-backdrop').classList.remove('open');
}

async function loadPhoneSelectors() {
  try {
    const brands = await api('/phones/meta/brands');
    const brandNode = $('#phone-brand');
    const current = brandNode.value;
    brandNode.innerHTML =
      `<option value="">All brands</option>` +
      brands.map((b) => `<option value="${esc(b.brand)}">${esc(b.brand)} (${b.phones})</option>`).join('');
    brandNode.value = current;
  } catch { /* optional */ }
}

const currentView = () => (location.hash || '#phones').slice(1).split('?')[0];
function go(view) { location.hash = view; }

async function renderView(view) {
  if (!VIEW_META[view]) view = 'phones';
  $$('.view').forEach((node) => node.classList.toggle('active', node.id === `view-${view}`));
  $$('.nav-item').forEach((node) => node.classList.toggle('active', node.dataset.view === view));
  const [title, subtitle] = VIEW_META[view];
  $('#page-title').textContent = title;
  $('#page-sub').textContent = subtitle;
  $('#sidebar').classList.remove('open');

  if (view === 'recommend') {
    if (state.lastRecommend) {
      restoreRecommendView();
      return;
    }
    setFlowStep(1);
    return;
  }
  if (view === 'phones') return renderPhones();
  if (view === 'evidence') return renderEvidence();
  if (view === 'evaluation') return renderEvaluation();
}

async function refreshAll() {
  state.features = [];
  await Promise.all([loadHealth(), loadBadges(), loadPhoneSelectors()]);
  await renderView(currentView());
}

async function init() {
  try {
    state.aspects = await api('/aspects');
  } catch {
    state.aspects = ['battery', 'camera', 'display', 'performance', 'design', 'price'].map((a) => ({
      aspect: a, label: a,
    }));
  }

  renderWeightControls();
  await Promise.all([loadHealth(), loadBadges(), loadPhoneSelectors()]);

  $$('.nav-item').forEach((node) => { node.onclick = () => go(node.dataset.view); });
  window.addEventListener('hashchange', () => renderView(currentView()));
  $('#btn-refresh').onclick = refreshAll;
  $('#btn-theme').onclick = toggleTheme;
  initTheme();
  $('#menu-toggle').onclick = () => $('#sidebar').classList.toggle('open');
  $('#btn-recommend').onclick = runRecommend;
  $$('input[name="rank-method"]').forEach((radio) => {
    radio.onchange = () => {
      if (radio.checked) state.rankingMethod = radio.value;
    };
  });
  const checked = $('input[name="rank-method"]:checked');
  if (checked) state.rankingMethod = checked.value;

  let searchTimer;
  $('#phone-search').oninput = () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(renderPhones, 280);
  };
  $('#phone-brand').onchange = renderPhones;

  $('#drawer-close').onclick = closeDrawer;
  $('#drawer-backdrop').onclick = closeDrawer;
  $('#lightbox-close').onclick = closeImageLightbox;
  $('#lightbox').onclick = (e) => {
    if (e.target === $('#lightbox') || e.target === $('#lightbox-img')) closeImageLightbox();
  };
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if ($('#lightbox') && !$('#lightbox').hidden) {
      closeImageLightbox();
      return;
    }
    closeDrawer();
  });

  await renderView(currentView());
}

window.go = go;
window.openPhone = openPhone;
window.openImageLightbox = openImageLightbox;
window.closeImageLightbox = closeImageLightbox;
window.phoneImageFailed = phoneImageFailed;
init();
