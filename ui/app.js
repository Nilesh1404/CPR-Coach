const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const post = (url, body={}) => fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)}).then(r=>r.json());

const STATE_DESC = {
  none:"Press Start to open the microphone, or type below to try it without audio.",
  checking:"Before CPR: is the person responding? breathing?",
  compressions:"The bystander is pushing on the chest. Beat is on.",
  confirming:"The coach asked “breathing normally now?” Waiting for a clear yes or no. Keep pushing.",
  stopped:"A confirmed “yes, breathing normally”. The coach told them to stop pushing."
};
let cfg = null, beatTimer = null, running = false, mode = "vad", lastState = null;

/* ---------- tabs ---------- */
document.querySelectorAll(".tabs button").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach(x => x.setAttribute("aria-selected", x === b));
  document.querySelectorAll(".view").forEach(v => v.hidden = v.id !== "view-" + b.dataset.view);
}));

/* ---------- network pill (for the Wi-Fi-off demo) ---------- */
function net(){
  const p = $("net");
  if (navigator.onLine){ p.textContent = "Online · turn Wi-Fi off for the demo"; p.className = "pill bad"; }
  else { p.textContent = "Offline ✓"; p.className = "pill ok"; }
}
addEventListener("online", net); addEventListener("offline", net); net();

/* ---------- live coach rendering ---------- */
function setStatus(s){ const p = $("status"); p.dataset.s = s; p.textContent = s; }

function setState(state, beat, lang){
  lastState = state;
  $("stateband").dataset.state = state || "none";
  $("state-name").textContent = state || "not started";
  $("state-desc").textContent = STATE_DESC[state || "none"];
  if (lang) $("lang-chip").textContent = "reply language: " + lang;
  document.querySelectorAll("#map span").forEach(s => s.classList.toggle("cur", s.dataset.st === state));
  setBeat(beat);
}

function setBeat(on){
  $("bpm").textContent = on ? "110 / min" : "beat off";
  $("bpm").classList.toggle("on", !!on);
  $("orb").classList.toggle("on", !!on);
  if (on && !beatTimer){
    beatTimer = setInterval(() => { const o = $("orb"); o.classList.add("tick"); setTimeout(() => o.classList.remove("tick"), 110); }, 60000/110);
  } else if (!on && beatTimer){ clearInterval(beatTimer); beatTimer = null; }
}

function addMsg(html){
  $("empty").hidden = true;
  const f = $("feed");
  f.insertAdjacentHTML("beforeend", html);
  f.scrollTop = f.scrollHeight;
}

function heardHTML(e){
  const meta = e.source === "voice"
    ? `speech-to-text ${Math.round(e.stt_ms)} ms · confidence ${e.logprob} · ${e.audio_s}s audio · ${e.stt_lang||""}`
    : "typed";
  return `<div class="msg you"><span class="who">Bystander</span><div class="bubble">${e.text ? esc(e.text) : "<i>(nothing understood)</i>"}</div><span class="meta">${esc(meta)}</span></div>`;
}

function replyHTML(e){
  const lines = e.lines.map(l => `<div class="line"><span class="lid ${l.id==="L10"?"stop":e.escalate&&l.id==="L11"?"esc":""}">${l.id}</span><span class="tx">${esc(l.text)}${l.text!==l.en?`<span class="en">${esc(l.en)}</span>`:""}</span></div>`).join("");
  const meta = e.intent ? `intent ${e.intent}${e.escalate?" · escalate":""} · ${e.nlu_ms} ms` : "opening line";
  return `<div class="msg coach"><span class="who">Coach</span><div class="bubble">${lines}</div><span class="meta">${esc(meta)}</span></div>`;
}

function showDecision(e, heard){
  if (!e.intent) return;
  $("dec-empty").hidden = true; $("dec-body").hidden = false;
  if (heard) $("dec-heard").textContent = heard.text ? `“${heard.text}”` : "(nothing understood)";
  $("dec-intent").textContent = e.intent;
  $("dec-near").textContent = e.nearest ? `“${e.nearest}”` : "none";
  const sc = Math.max(0, Math.min(1, e.score || 0));
  $("dec-score").style.width = (sc*100) + "%";
  $("dec-score-t").textContent = "match " + (e.score ?? 0).toFixed(2);
  const f = e.flags || {};
  const chips = [["neg","negation word"],["doubt","doubt word"],["gasp","gasp word"],["breath","breath word"],["yes_first","starts with yes"]]
    .filter(([k]) => f[k]).map(([k,t]) => `<span class="chip ${k==="yes_first"?"yes":k}">${t}</span>`);
  $("dec-flags").innerHTML = chips.join("") || `<span class="chip">no safety words</span>`;
  const wasConfirming = e.lines.some(l => l.id === "L10") || (heard && e.intent === "breathing_normal" && (e.state === "confirming" || e.state === "stopped"));
  if (wasConfirming){
    const lp = heard && heard.logprob;
    const checks = [
      ["state is confirming", true],
      ["intent is breathing_normal", e.intent === "breathing_normal"],
      ["no negation / doubt / gasp word", !f.neg && !f.doubt && !f.gasp],
      ["speech-to-text confident", lp == null || lp >= -0.8]
    ];
    $("dec-gate").innerHTML = `<b>Stop gate</b>` + checks.map(([t,ok]) => `<div><span class="${ok?"ok":"no"}">${ok?"✓":"✗"}</span> ${t}</div>`).join("")
      + `<div>${e.lines.some(l=>l.id==="L10") ? "<span class='ok'>→ L10 stop pushing</span>" : "<span class='no'>→ keep pushing</span>"}</div>`;
    $("dec-gate").hidden = false;
  } else $("dec-gate").hidden = true;
  const t = [];
  if (heard && heard.stt_ms != null) t.push(`speech-to-text ${Math.round(heard.stt_ms)} ms`);
  if (e.nlu_ms != null) t.push(`understanding + decision ${e.nlu_ms} ms`);
  $("dec-time").textContent = t.join(" · ");
}

let lastHeard = null;
function onEvent(e){
  switch (e.type){
    case "status": setStatus(e.status); break;
    case "level": $("meter").style.width = Math.min(100, Math.sqrt(e.rms) * 260) + "%"; break;
    case "heard": lastHeard = e; addMsg(heardHTML(e)); break;
    case "reply": addMsg(replyHTML(e)); setState(e.state, e.beat, e.lang); showDecision(e, lastHeard); break;
    case "system": addMsg(`<div class="sys ${e.error?"err":""}">${esc(e.text)}</div>`); if (/^Stopped|Could not start/.test(e.text)) setRunning(false); break;
    case "mode": setMode(e.mode, false); break;
  }
}

function setRunning(on){
  running = on;
  $("start").disabled = on; $("stop").disabled = !on;
  $("model").disabled = on; $("stt-lang").disabled = on;
}

function setMode(m, send=true){
  mode = m;
  document.querySelectorAll(".seg button").forEach(b => b.setAttribute("aria-checked", b.dataset.mode === m));
  $("ptt").hidden = m !== "ptt";
  if (send) post("/api/mode", {mode:m});
}

/* ---------- controls ---------- */
$("start").onclick = () => { setRunning(true); post("/api/start", {stt:$("model").value, lang:$("stt-lang").value, mode}); };
$("stop").onclick = () => { post("/api/stop"); };
$("reset").onclick = () => { $("feed").querySelectorAll(".msg,.sys").forEach(n => n.remove()); $("empty").hidden = false; setState(null, false); $("dec-body").hidden = true; $("dec-empty").hidden = false; post("/api/reset"); };
document.querySelectorAll(".seg button").forEach(b => b.onclick = () => setMode(b.dataset.mode));
$("composer").onsubmit = ev => { ev.preventDefault(); const t = $("say").value.trim(); if (!t) return; $("say").value = ""; post("/api/say", {text:t}); };

/* push to talk: mouse/touch hold or Space */
const ptt = down => { $("ptt").classList.toggle("down", down); post("/api/ptt", {down}); };
$("ptt").addEventListener("pointerdown", e => { e.preventDefault(); ptt(true); });
["pointerup","pointerleave","pointercancel"].forEach(n => $("ptt").addEventListener(n, () => $("ptt").classList.contains("down") && ptt(false)));
let spaceDown = false;
addEventListener("keydown", e => { if (e.code==="Space" && mode==="ptt" && !spaceDown && document.activeElement.tagName!=="INPUT"){ e.preventDefault(); spaceDown = true; ptt(true); }});
addEventListener("keyup", e => { if (e.code==="Space" && spaceDown){ spaceDown = false; ptt(false); }});

/* ---------- text test ---------- */
const EXAMPLES = [
  ["saans nahi le raha","checking"],["haan saans le raha hai","confirming"],["shayad saans le raha","confirming"],
  ["haanf raha hai","confirming"],["haath thak gaye bhai","compressions"],["haath kahan rakhu","compressions"],
  ["हो श्वास घेत आहे","confirming"],["नाही, श्वास घेत नाहीये","confirming"],["paani pila du kya","compressions"]
];
$("t-examples").innerHTML = EXAMPLES.map(([t,s],i) => `<button type="button" class="chip" data-i="${i}">${esc(t)}</button>`).join("");
$("t-examples").onclick = ev => { const b = ev.target.closest("button"); if (!b) return; const [t,s] = EXAMPLES[b.dataset.i]; $("t-text").value = t; $("t-state").value = s; runTest(); };
$("testform").onsubmit = ev => { ev.preventDefault(); runTest(); };

async function runTest(){
  const r = await post("/api/test", {text:$("t-text").value, state:$("t-state").value});
  const f = r.flags || {};
  const chips = [["neg","negation"],["doubt","doubt"],["gasp","gasp"],["breath","breath"],["yes_first","starts with yes"]].filter(([k])=>f[k]).map(([k,t])=>`<span class="chip ${k==="yes_first"?"yes":k}">${t}</span>`).join("") || `<span class="chip">no safety words</span>`;
  $("t-result").innerHTML = `
    <div class="kv"><span>Intent</span><span class="big mono">${esc(r.intent)}</span></div>
    <div class="kv"><span>Matched example</span><span class="mono">${r.nearest?`“${esc(r.nearest)}”`:"none"} · ${(r.score??0).toFixed(2)}${r.knn_intent&&r.knn_intent!==r.intent?` (matching said ${esc(r.knn_intent)}, rules changed it)`:""}</span></div>
    <div class="chips">${chips}</div>
    <div class="bubble" style="background:var(--soft);border-radius:12px">${r.lines.map(l=>`<div class="line"><span class="lid ${l.id==="L10"?"stop":""}">${l.id}</span><span class="tx">${esc(l.text)}${l.text!==l.en?`<span class="en">${esc(l.en)}</span>`:""}</span></div>`).join("")}</div>
    <div class="kv"><span>State after</span><span class="mono">${esc(r.state_after||"")}</span></div>
    <div class="kv"><span>Escalate</span><span>${r.escalate?"yes":"no"}</span></div>
    <div class="timing">${r.ms} ms${r.error?" · error: "+esc(r.error):""}</div>`;
}

$("t-file").onchange = async ev => {
  const file = ev.target.files[0]; if (!file) return;
  let data;
  try { data = JSON.parse(await file.text()); }
  catch(e){ $("t-batch").innerHTML = `<tr><td>That file isn’t valid JSON: ${esc(e.message)}</td></tr>`; return; }
  const items = data.items || [];
  const res = await post("/api/batch", {items});
  const byId = Object.fromEntries(items.map(i => [i.id, i]));
  $("t-batch").innerHTML = `<thead><tr><th>id</th><th>state</th><th>text</th><th>intent</th><th>reply</th><th>escalate</th><th>ms</th></tr></thead><tbody>` +
    res.results.map(r => `<tr><td class="mono">${esc(r.id)}</td><td>${esc(byId[r.id]?.state)}</td><td>${esc(byId[r.id]?.text)}</td><td class="mono">${esc(r.intent)}</td><td class="mono">${r.reply.join(" ")}</td><td>${r.escalate?"yes":""}</td><td>${r.ms}</td></tr>`).join("") + "</tbody>";
  const a = $("t-download");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(res, null, 2)], {type:"application/json"}));
  a.hidden = false;
};

/* ---------- boot ---------- */
(async function boot(){
  cfg = await fetch("/api/config").then(r => r.json());
  $("model").innerHTML = cfg.models.map(([v, label], i) => `<option value="${v}" ${i===0?"selected":""}>${label}</option>`).join("");
  const s = await fetch("/api/state").then(r => r.json());
  setStatus(s.status); setRunning(s.running); setMode(s.mode, false);
  if (s.stt) $("model").value = s.stt;
  s.history.forEach(onEvent);
  setState(s.started ? s.state : null, s.beat, s.lang);
  const es = new EventSource("/api/events?after=" + s.seq);
  es.onmessage = m => onEvent(JSON.parse(m.data));
  es.onerror = () => setStatus("disconnected");
})();
