const DEFAULT_SERVER_URL = `${window.location.protocol}//${window.location.hostname}:8000`;
const SERVER_URL = new URLSearchParams(window.location.search).get("server") || DEFAULT_SERVER_URL;

const chatEl = document.getElementById("chat");
const formEl = document.getElementById("composer");
const inputEl = document.getElementById("input");
const sendEl = document.getElementById("send");
const serverEl = document.getElementById("server-url");

serverEl.textContent = SERVER_URL;

function addMessage(role, text, isTyping = false) {
  const msg = document.createElement("div");
  msg.className = `msg ${role}` + (isTyping ? " typing" : "");

  const meta = document.createElement("div");
  meta.className = "msg-meta";

  const badge = document.createElement("span");
  badge.className = "msg-role";
  badge.textContent = role === "user" ? "You" : "Assistant";

  const time = document.createElement("span");
  time.className = "msg-time";
  time.textContent = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

  meta.append(badge, time);

  const body = document.createElement("div");
  body.className = "msg-body";
  body.textContent = text;
  msg.append(meta, body);

  chatEl.appendChild(msg);
  chatEl.scrollTop = chatEl.scrollHeight;
  return msg;
}

function streamText(targetEl, text) {
  if (!text) {
    targetEl.classList.remove("typing");
    targetEl.textContent = "(empty response)";
    return;
  }

  const tokens = text.split(/\s+/);
  let index = 0;

  const timer = setInterval(() => {
    targetEl.textContent = tokens.slice(0, index + 1).join(" ");
    chatEl.scrollTop = chatEl.scrollHeight;
    index += 1;
    if (index >= tokens.length) {
      clearInterval(timer);
      targetEl.classList.remove("typing");
    }
  }, 35);
}

function normalizeOutput(payload) {
  const raw = payload && payload.output ? payload.output.raw ?? payload.output : payload;
  if (typeof raw === "string") return raw;
  try {
    return JSON.stringify(raw, null, 2);
  } catch (err) {
    return String(raw);
  }
}

async function sendQuery(query) {
  const response = await fetch(`${SERVER_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText} - ${text}`);
  }

  const data = await response.json();
  return {
    answer: normalizeOutput(data),
    citations: Array.isArray(data.citations) ? data.citations : [],
  };
}

function renderCitations(msgEl, citations) {
  if (!citations || citations.length === 0) return;

  const wrapper = document.createElement("details");
  wrapper.className = "citations";
  wrapper.open = false;

  const title = document.createElement("summary");
  title.className = "citations-title";
  title.textContent = `Sources (${citations.length})`;
  wrapper.appendChild(title);

  const list = document.createElement("div");
  list.className = "citation-list";
  citations.forEach((item) => {
    const card = document.createElement("div");
    card.className = "citation-item";
    const source = item.source || "unknown";
    const pageStart = item.page_start;
    const pageEnd = item.page_end;
    let pageLabel = "page ?";
    if (pageStart && pageEnd && pageStart !== pageEnd) {
      pageLabel = `pages ${pageStart}-${pageEnd}`;
    } else if (pageStart) {
      pageLabel = `page ${pageStart}`;
    }
    const heading = document.createElement("div");
    heading.className = "citation-source";
    heading.textContent = `${source} (${pageLabel})`;
    card.appendChild(heading);

    if (item.snippet) {
      const snippet = document.createElement("div");
      snippet.className = "citation-snippet";
      snippet.textContent = item.snippet;
      card.appendChild(snippet);
    }

    list.appendChild(card);
  });

  wrapper.appendChild(list);
  msgEl.appendChild(wrapper);
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = inputEl.value.trim();
  if (!query) return;

  addMessage("user", query);
  inputEl.value = "";
  inputEl.focus();

  sendEl.disabled = true;
  const assistantEl = addMessage("assistant", "", true);

  try {
    const { answer, citations } = await sendQuery(query);
    const bodyEl = assistantEl.querySelector(".msg-body") || assistantEl;
    streamText(bodyEl, answer);
    renderCitations(assistantEl, citations);
  } catch (error) {
    assistantEl.classList.remove("typing");
    const bodyEl = assistantEl.querySelector(".msg-body") || assistantEl;
    bodyEl.textContent = `Error: ${error.message}`;
  } finally {
    sendEl.disabled = false;
  }
});

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});
