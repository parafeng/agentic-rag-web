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
  msg.textContent = text;
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

function normalizeOutput(data) {
  const raw = data && data.output ? data.output.raw ?? data.output : data;
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
  return normalizeOutput(data);
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
    const answer = await sendQuery(query);
    streamText(assistantEl, answer);
  } catch (error) {
    assistantEl.classList.remove("typing");
    assistantEl.textContent = `Error: ${error.message}`;
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
