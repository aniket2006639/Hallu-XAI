// Runs inside the chat page. It can (1) read the latest AI answer and (2) highlight flagged claims.
// NOTE: chat sites change their HTML often. If auto-detection stops working, the fallbacks are:
// select the text yourself, or paste it into the side panel.

const ASSISTANT_SELECTORS = [
  '[data-message-author-role="assistant"]',   // ChatGPT
  '.font-claude-message',                      // Claude
  '[data-testid="assistant-message"]',
  'message-content',                           // Gemini
  '.model-response-text',
];
const USER_SELECTORS = [
  '[data-message-author-role="user"]',         // ChatGPT
  '[data-testid="user-message"]',              // Claude
  'user-query',                                // Gemini
  '.user-query-text',
];

function lastText(selectors) {
  for (const sel of selectors) {
    const nodes = document.querySelectorAll(sel);
    if (nodes.length) return nodes[nodes.length - 1].innerText.trim();
  }
  return "";
}

function getPageText() {
  const selected = window.getSelection().toString().trim();
  if (selected) return { answer: selected, question: "", source: "your selection" };
  return { answer: lastText(ASSISTANT_SELECTORS), question: lastText(USER_SELECTORS), source: "the last AI answer" };
}

// ---- highlighting, using the CSS Custom Highlight API (does not modify the page's DOM) ----
const style = document.createElement("style");
style.textContent = `
  ::highlight(hallu-flag) { background-color: rgba(229, 72, 77, 0.35); }
  ::highlight(hallu-warn) { background-color: rgba(245, 166, 35, 0.35); }
`;
document.documentElement.appendChild(style);

function escapeRegex(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

// Build a whitespace-tolerant regex so line breaks / double spaces on the page don't break matching.
function looseRegex(text) {
  return new RegExp(text.trim().split(/\s+/).map(escapeRegex).join("\\s+"), "i");
}

function findRanges(claim) {
  const ranges = [];
  const full = looseRegex(claim);
  const prefix = looseRegex(claim.trim().split(/\s+/).slice(0, 8).join(" "));
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    const text = node.textContent;
    let m = text.match(full);
    let end = m ? m.index + m[0].length : -1;
    if (!m) {                       // claim is split across formatting (bold, links...): mark from its start
      m = text.match(prefix);
      end = m ? text.length : -1;
    }
    if (m) {
      const r = new Range();
      r.setStart(node, m.index);
      r.setEnd(node, end);
      ranges.push(r);
    }
  }
  return ranges;
}

function applyHighlights(claims) {
  if (!("highlights" in CSS)) return 0;
  CSS.highlights.clear();
  const flag = new Highlight(), warn = new Highlight();
  let count = 0;
  for (const c of claims) {
    const target = c.verdict === "likely_hallucinated" ? flag : c.verdict === "uncertain" ? warn : null;
    if (!target) continue;
    for (const r of findRanges(c.text)) { target.add(r); count++; }
  }
  CSS.highlights.set("hallu-flag", flag);
  CSS.highlights.set("hallu-warn", warn);
  return count;
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "GET_PAGE_TEXT") sendResponse(getPageText());
  if (msg.type === "HIGHLIGHT") sendResponse({ highlighted: applyHighlights(msg.claims || []) });
  if (msg.type === "CLEAR_HIGHLIGHT") { if ("highlights" in CSS) CSS.highlights.clear(); sendResponse({}); }
  return true;
});
