const $ = (id) => document.getElementById(id);

// Build DOM with textContent only - page text and evidence are untrusted, never use innerHTML.
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function setStatus(message, isError = false) {
  const status = $("status");
  status.textContent = message;
  status.className = "status" + (isError ? " error" : "");
}

async function activeTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

async function readPage() {
  const tab = await activeTab();
  if (!tab || !tab.id) throw new Error("No active tab found.");
  try {
    return { tab, page: await chrome.tabs.sendMessage(tab.id, { type: "GET_PAGE_TEXT" }) };
  } catch (e) {
    throw new Error("Can't read this page. Open a ChatGPT / Claude / Gemini chat and reload it once, " +
                    "or paste the text below.");
  }
}

// ---------- rendering ----------
const VERDICT_LABEL = {
  likely_hallucinated: "Likely hallucinated",
  uncertain: "Uncertain",
  supported: "Supported",
};

function shapBars(contributions) {
  const wrap = el("div");
  wrap.appendChild(el("div", "shap-title", "Why (SHAP): what pushed this score"));
  const maxAbs = Math.max(...contributions.map((c) => Math.abs(c.shap)), 0.001);
  for (const c of contributions) {
    const row = el("div", "shap-row");
    row.title = c.meaning;
    row.appendChild(el("span", "", c.feature.replace(/_/g, " ")));
    const track = el("div", "bar-track");
    const bar = el("div", "bar " + (c.shap > 0 ? "pos" : "neg"));
    bar.style.width = (Math.abs(c.shap) / maxAbs) * 50 + "%";
    track.appendChild(bar);
    row.appendChild(track);
    row.appendChild(el("span", "", (c.shap > 0 ? "+" : "") + c.shap.toFixed(2)));
    wrap.appendChild(row);
  }
  wrap.appendChild(el("div", "legend", "red → toward hallucinated · green → toward supported"));
  return wrap;
}

function renderClaim(item) {
  const card = el("div", "claim " + item.verdict);
  card.appendChild(el("p", "claim-text", item.claim));

  const line = el("div");
  line.appendChild(el("span", "chip " + item.verdict, VERDICT_LABEL[item.verdict]));
  line.appendChild(el("span", "", `risk score ${(item.score * 100).toFixed(0)}%`));
  card.appendChild(line);

  card.appendChild(el("p", "reason", item.explanation.reason));
  card.appendChild(shapBars(item.explanation.contributions));

  const details = el("details", "evidence");
  details.appendChild(el("summary", "", `Evidence used (${item.evidence.length})`));
  if (!item.evidence.length) details.appendChild(el("p", "", "No evidence was found for this claim."));
  for (const e of item.evidence) {
    const p = el("p");
    p.appendChild(el("span", "src", e.source + ": "));
    p.appendChild(document.createTextNode(e.text));
    details.appendChild(p);
  }
  card.appendChild(details);
  return card;
}

function render(data) {
  const box = $("results");
  box.replaceChildren();

  const summary = el("div", "summary");
  summary.appendChild(el("strong", "", data.overall.summary));
  for (const w of data.meta.warnings || []) summary.appendChild(el("div", "warning", "⚠ " + w));
  summary.appendChild(el("div", "legend", `${data.meta.latency_ms} ms`));
  box.appendChild(summary);

  for (const item of data.claims) box.appendChild(renderClaim(item));
}

// ---------- main action ----------
$("analyze").addEventListener("click", async () => {
  const button = $("analyze");
  button.disabled = true;
  try {
    let question = "", answer = $("manual").value.trim(), source = "pasted text", tab = null;
    if (!answer) {
      setStatus("Reading the page…");
      const result = await readPage();
      tab = result.tab;
      ({ answer, question, source } = result.page);
    }
    if (!answer) {
      setStatus("No text found. Select some text on the page, or paste it below.", true);
      return;
    }

    setStatus(`Analyzing ${source}…`);
    const res = await chrome.runtime.sendMessage({ type: "ANALYZE", question, answer });
    if (!res || !res.ok) {
      setStatus((res && res.error) || "Backend not reachable. Is `uvicorn` running on port 8000?", true);
      return;
    }

    render(res.data);
    setStatus(`Done - analyzed ${source}.`);

    if (tab) {
      const claims = res.data.claims.map((c) => ({ text: c.claim, verdict: c.verdict }));
      chrome.tabs.sendMessage(tab.id, { type: "HIGHLIGHT", claims }).catch(() => {});
    }
  } catch (err) {
    setStatus(String(err.message || err), true);
  } finally {
    button.disabled = false;
  }
});

// ---------- backend status badge ----------
chrome.runtime.sendMessage({ type: "HEALTH" }, (res) => {
  const badge = $("backend-badge");
  if (res && res.ok) { badge.textContent = "backend online"; badge.className = "badge ok"; }
  else { badge.textContent = "backend offline"; badge.className = "badge bad"; }
});
