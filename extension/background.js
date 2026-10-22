// The relay between the side panel and the Python backend.
const BACKEND = "http://localhost:8000";

// Clicking the toolbar icon opens the side panel.
chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(console.error);

async function callBackend(path, options) {
  const response = await fetch(BACKEND + path, options);
  if (!response.ok) throw new Error(`Backend returned ${response.status}: ${await response.text()}`);
  return response.json();
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "ANALYZE") {
    callBackend("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: msg.question || "", answer: msg.answer }),
    })
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err.message || err) }));
    return true; // keep the message channel open for the async response
  }

  if (msg.type === "HEALTH") {
    callBackend("/health")
      .then((data) => sendResponse({ ok: true, data }))
      .catch(() => sendResponse({ ok: false }));
    return true;
  }
});
