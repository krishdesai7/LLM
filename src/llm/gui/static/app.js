"use strict";

// The chat page. Plain JS, no build step. Talks to the JSON/SSE API in app.py.

// Identifies this page to the server's heartbeat. Kept per tab, so a reload is
// still the same page and its replies are not mistaken for abandoned ones.
const OWNER = (() => {
  try {
    const saved = sessionStorage.getItem("llm-owner");
    if (saved) return saved;
    const fresh = crypto.randomUUID();
    sessionStorage.setItem("llm-owner", fresh);
    return fresh;
  } catch {
    return crypto.randomUUID();
  }
})();
const $ = (id) => document.getElementById(id);

const els = {
  list: $("chat-list"),
  newChat: $("new-chat"),
  model: $("model-label"),
  web: $("web-toggle"),
  thinking: $("thinking-toggle"),
  settingsOpen: $("settings-open"),
  flash: $("flash"),
  box: $("messages"),
  column: $("column"),
  composer: $("composer"),
  prompt: $("prompt"),
  send: $("send"),
  stop: $("stop"),
  dialog: $("settings"),
  form: $("settings-form"),
  provider: $("set-provider"),
  modelInput: $("set-model"),
  modelOptions: $("model-options"),
  system: $("set-system"),
  temperature: $("set-temperature"),
  problem: $("set-problem"),
  cancel: $("set-cancel"),
};

const state = {
  info: null, // /api/state
  chats: [], // sidebar summaries
  chat: null, // the open chat: {id, settings, running}; null for an unsaved new chat
  draft: { model: "", system: null, temperature: null, web: false }, // settings for a new chat
  turn: null, // Turn receiving live events for the open chat
  abort: null, // AbortController of the stream shown in this page
  sending: false, // a prompt is being sent but its stream hasn't started
  armed: null, // chat id whose delete button has been clicked once
};

// -- markdown ---------------------------------------------------------------------

marked.setOptions({ gfm: true, breaks: false });

DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  if (node.tagName === "A") {
    node.setAttribute("target", "_blank");
    node.setAttribute("rel", "noopener noreferrer");
  }
});

// Model output may contain text from fetched web pages. Anything that could load
// something remote (images, media, frames) or restyle the page is dropped.
const PURIFY = {
  FORBID_TAGS: ["img", "picture", "source", "video", "audio", "iframe", "form", "input", "button", "style", "link"],
  FORBID_ATTR: ["style", "srcset"],
};

function renderMarkdown(text) {
  return DOMPurify.sanitize(marked.parse(text), PURIFY);
}

// -- small helpers ----------------------------------------------------------------

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

class ApiError extends Error {
  constructor(status, detail) {
    super(detail);
    this.status = status;
  }
}

async function request(method, path, body) {
  const response = await fetch(path, {
    method,
    headers: body === undefined ? {} : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) throw await apiError(response);
  return response;
}

async function apiError(response) {
  let detail = `${response.status} ${response.statusText}`;
  try {
    const data = await response.json();
    if (data.detail) detail = data.detail;
  } catch {
    // keep the status line
  }
  return new ApiError(response.status, detail);
}

const getJSON = async (path) => (await request("GET", path)).json();

function flash(message) {
  els.flash.textContent = message;
  els.flash.hidden = !message;
  if (message) setTimeout(() => { if (els.flash.textContent === message) flash(""); }, 6000);
}

function seconds(value) {
  if (!value) return "";
  return value < 1 ? "<1s" : `${Math.round(value)}s`;
}

function compact(text, limit = 90) {
  const flat = String(text).split(/\s+/).join(" ");
  return flat.length <= limit ? flat : `${flat.slice(0, limit)}…`;
}

function describeResult(content) {
  const lines = content ? content.split("\n").length : 0;
  return `${content.length} chars, ${lines} lines`;
}

// Keep following the bottom of the chat unless the reader has scrolled up.
function keepingPinned(change) {
  const box = els.box;
  const pinned = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
  change();
  if (pinned) box.scrollTop = box.scrollHeight;
}

// -- one assistant turn -------------------------------------------------------------

class Turn {
  constructor(parent) {
    this.root = el("div", "turn");
    parent.append(this.root);
    this.think = null; // the thinking block still receiving text
    this.answer = null; // the answer segment still receiving text
    this.tool = null; // the last tool row, awaiting its result
  }

  handle(event) {
    switch (event.type) {
      case "reasoning": return this.reasoning(event.text);
      case "text": return this.text(event.text);
      case "tool_call": return this.toolCall(event.text);
      case "tool_result": return this.toolResult(event.text);
      case "notice": return void this.root.append(el("div", "notice", event.text));
      case "usage":
        return void this.root.append(el("div", "usage", `${event.prompt} in / ${event.completion} out`));
      case "error": return this.error(event.text);
      case "done": return this.finish();
    }
  }

  reasoning(text) {
    this.closeAnswer();
    if (!this.think) {
      const details = el("details", "think");
      details.open = true;
      const summary = el("summary", "", "Thinking…");
      const body = el("div", "body");
      details.append(summary, body);
      this.root.append(details);
      this.think = { details, summary, body, started: performance.now() };
    }
    this.think.body.textContent += text;
  }

  closeThinking() {
    const think = this.think;
    if (!think) return;
    const took = (performance.now() - think.started) / 1000;
    think.summary.textContent = `Thinking · ${seconds(took)}`;
    think.details.open = false; // the answer is what you came for
    this.think = null;
  }

  text(text) {
    this.closeThinking();
    if (!this.answer) {
      this.answer = { div: el("div", "answer"), text: "", scheduled: false };
      this.root.append(this.answer.div);
    }
    const answer = this.answer;
    answer.text += text;
    if (!answer.scheduled) {
      answer.scheduled = true;
      requestAnimationFrame(() => {
        answer.scheduled = false;
        keepingPinned(() => { answer.div.innerHTML = renderMarkdown(answer.text); });
      });
    }
  }

  closeAnswer() {
    if (!this.answer) return;
    this.answer.div.innerHTML = renderMarkdown(this.answer.text);
    this.answer = null;
  }

  toolCall(text) {
    this.closeThinking();
    this.closeAnswer();
    const row = el("div", "tool", text);
    this.root.append(row);
    this.tool = row;
    return row;
  }

  toolResult(text) {
    if (!this.tool) return;
    this.tool.append(el("span", "result", `↳ ${text}`));
    this.tool = null;
  }

  error(text) {
    this.closeThinking();
    this.root.append(el("div", "error", text));
  }

  finish() {
    this.closeThinking();
    this.closeAnswer();
  }

  // For chats loaded from disk.
  addThinking(text, took) {
    const details = el("details", "think");
    details.append(el("summary", "", took ? `Thinking · ${seconds(took)}` : "Thinking"), el("div", "body", text));
    this.root.append(details);
  }

  addAnswer(text) {
    const div = el("div", "answer");
    div.innerHTML = renderMarkdown(text);
    this.root.append(div);
  }
}

function renderMessages(messages) {
  els.column.replaceChildren();
  if (!messages.length) {
    els.column.append(el("div", "empty", "What's on your mind?"));
    return;
  }
  let turn = null;
  const calls = new Map(); // tool_call_id -> its row
  for (const message of messages) {
    if (message.role === "user") {
      els.column.append(el("div", "user", message.content));
      turn = null;
    } else if (message.role === "assistant") {
      turn ??= new Turn(els.column);
      if (message.reasoning) turn.addThinking(message.reasoning, message.reasoning_seconds);
      for (const call of message.tool_calls) {
        calls.set(call.id, turn.toolCall(`${call.name}(${compact(call.arguments)})`));
      }
      if (message.content) turn.addAnswer(message.content);
    } else if (message.role === "tool") {
      const row = calls.get(message.tool_call_id);
      if (row) row.append(el("span", "result", `↳ ${describeResult(message.content)}`));
    }
  }
  els.box.scrollTop = els.box.scrollHeight;
}

// -- streaming ----------------------------------------------------------------------

async function readEvents(body, onEvent) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let sawDone = false;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop();
    for (const block of blocks) {
      const data = block.split(/\r?\n/).filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trimStart()).join("\n");
      if (!data) continue;
      const event = JSON.parse(data);
      if (event.type === "done") sawDone = true;
      onEvent(event);
    }
  }
  return sawDone;
}

// Show a reply's events in the open chat. Leaving the chat only detaches this
// reader; the reply carries on server-side.
// `accepted` runs once the server has taken the request; `rejected(error)` runs instead if it
// never did (a 4xx/5xx, or no connection), so the caller can undo what it showed optimistically.
async function follow(path, body, { accepted = null, rejected = null } = {}) {
  const controller = new AbortController();
  state.abort = controller;
  const turn = state.turn;
  let started = false;
  setBusy(true);
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok) throw await apiError(response);
    started = true;
    if (accepted) accepted();
    const finished = await readEvents(response.body, (event) => keepingPinned(() => turn.handle(event)));
    if (!finished) keepingPinned(() => turn.error("Connection lost. The reply may still be running; reopen this chat to reattach."));
  } catch (error) {
    if (error.name === "AbortError") return null;
    if (error instanceof ApiError && path.endsWith("/attach") && error.status === 404) return error;
    if (!started && rejected) {
      rejected(error);
      return null;
    }
    keepingPinned(() => turn.error(error.message));
  } finally {
    if (state.abort === controller) {
      state.abort = null;
      setBusy(false);
      refreshChats();
    }
  }
  return null;
}

function detach() {
  const controller = state.abort;
  state.abort = null;
  if (controller) controller.abort();
  setBusy(false);
}

function setBusy(busy) {
  els.send.hidden = busy;
  els.stop.hidden = !busy;
}

// -- chats --------------------------------------------------------------------------

async function openChat(id) {
  detach();
  let chat;
  try {
    chat = await getJSON(`/api/chats/${encodeURIComponent(id)}`);
  } catch (error) {
    flash(error.message);
    return refreshChats();
  }
  state.chat = { id: chat.id, settings: chat.settings, running: chat.running };
  state.draft = { ...state.draft, model: chat.settings.model };
  updateHeader();
  renderSidebar();

  let messages = chat.messages;
  if (chat.running) {
    // The reply's own events are replayed from the start, so show only what
    // came before the prompt that started it.
    const lastUser = messages.map((m) => m.role).lastIndexOf("user");
    messages = messages.slice(0, lastUser + 1);
  }
  renderMessages(messages);
  state.turn = null;
  if (!chat.running) return;

  state.turn = new Turn(els.column);
  const missed = await follow(`/api/chats/${encodeURIComponent(id)}/attach`, { after: 0, owner: OWNER });
  if (missed) return openChat(id); // it finished in the meantime: show it from disk
}

function newChat() {
  detach();
  state.chat = null;
  state.turn = null;
  renderMessages([]);
  updateHeader();
  renderSidebar();
  els.prompt.focus();
}

async function createChat() {
  const settings = state.draft;
  if (!settings.model) {
    openSettings("Pick a model first.");
    throw new Error("no model");
  }
  const { id } = await (await request("POST", "/api/chats", settings)).json();
  state.chat = { id, settings: { ...settings }, running: false };
  return id;
}

async function send() {
  const prompt = els.prompt.value.trim();
  // `sending` covers the gap before follow() sets `abort`, so a second Enter can't start a second chat.
  if (!prompt || state.abort || state.sending) return;
  state.sending = true;
  try {
    await sendPrompt(prompt);
  } finally {
    state.sending = false;
  }
}

async function sendPrompt(prompt) {
  try {
    if (!state.chat) await createChat();
  } catch (error) {
    if (error.message !== "no model") flash(error.message);
    return;
  }
  if (!els.column.querySelector(".user, .turn")) els.column.replaceChildren();
  const bubble = el("div", "user", prompt);
  els.column.append(bubble);
  const turn = new Turn(els.column);
  state.turn = turn;
  els.box.scrollTop = els.box.scrollHeight;
  await follow(`/api/chats/${encodeURIComponent(state.chat.id)}/reply`, { prompt, owner: OWNER }, {
    // The text stays in the box until the server has it, so a refusal loses nothing.
    accepted() {
      if (els.prompt.value.trim() === prompt) els.prompt.value = "";
      autosize();
    },
    rejected(error) {
      bubble.remove();
      turn.root.remove();
      if (state.turn === turn) state.turn = null;
      flash(error.message);
    },
  });
}

async function stop() {
  if (!state.chat) return;
  try {
    await request("POST", `/api/chats/${encodeURIComponent(state.chat.id)}/stop`);
  } catch (error) {
    flash(error.message);
  }
}

// -- sidebar --------------------------------------------------------------------------

async function refreshChats() {
  try {
    state.chats = await getJSON("/api/chats");
  } catch {
    return;
  }
  renderSidebar();
}

function groupOf(updated) {
  const startOfToday = new Date().setHours(0, 0, 0, 0) / 1000;
  const day = 86400;
  if (updated >= startOfToday) return "Today";
  if (updated >= startOfToday - day) return "Yesterday";
  if (updated >= startOfToday - 7 * day) return "Last 7 days";
  return "Older";
}

function renderSidebar() {
  const groups = new Map();
  for (const chat of state.chats) {
    const name = groupOf(chat.updated);
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push(chat);
  }
  const nodes = [];
  for (const [name, chats] of groups) {
    nodes.push(el("h4", "", name));
    for (const chat of chats) nodes.push(chatItem(chat));
  }
  els.list.replaceChildren(...nodes);
}

function chatItem(chat) {
  const item = el("div", `chat-item${state.chat?.id === chat.id ? " on" : ""}`);
  item.append(el("span", "title", chat.title));
  if (chat.running) item.append(el("span", "busy", "●"));
  const del = el("button", "del", state.armed === chat.id ? "Delete?" : "×");
  del.type = "button";
  del.title = "Delete this chat";
  if (state.armed === chat.id) del.classList.add("armed");
  del.addEventListener("click", (event) => {
    event.stopPropagation();
    removeChat(chat.id);
  });
  item.append(del);
  item.addEventListener("click", () => openChat(chat.id));
  return item;
}

async function removeChat(id) {
  if (state.armed !== id) {
    state.armed = id;
    renderSidebar();
    setTimeout(() => { if (state.armed === id) { state.armed = null; renderSidebar(); } }, 3000);
    return;
  }
  state.armed = null;
  try {
    await request("DELETE", `/api/chats/${encodeURIComponent(id)}`);
  } catch (error) {
    flash(error.message);
  }
  if (state.chat?.id === id) newChat();
  refreshChats();
}

// -- header and settings ---------------------------------------------------------------

const currentSettings = () => state.chat?.settings ?? state.draft;

function updateHeader() {
  const settings = currentSettings();
  els.model.textContent = settings.model || "Choose a model";
  els.web.textContent = `🌐 Web ${settings.web ? "on" : "off"}`;
  els.web.setAttribute("aria-pressed", String(Boolean(settings.web)));
  els.web.title = state.info ? state.info.backend : "";
}

async function changeSettings(next) {
  if (!state.chat) {
    state.draft = next;
    updateHeader();
    return;
  }
  await request("PUT", `/api/chats/${encodeURIComponent(state.chat.id)}/settings`, next);
  state.chat.settings = next;
  state.draft = { ...state.draft, model: next.model };
  updateHeader();
}

async function toggleWeb() {
  try {
    await changeSettings({ ...currentSettings(), web: !currentSettings().web });
  } catch (error) {
    flash(error.message);
  }
}

function toggleAllThinking() {
  const blocks = [...els.column.querySelectorAll("details.think")];
  const anyOpen = blocks.some((block) => block.open);
  for (const block of blocks) block.open = !anyOpen;
}

function openSettings(problem) {
  const settings = currentSettings();
  const [head, ...rest] = settings.model.split("/");
  const known = state.info.providers.some((p) => p.name === head);
  els.provider.replaceChildren(
    ...state.info.providers.map((p) => {
      const option = el("option", "", p.ready ? p.title : `${p.title} (no key)`);
      option.value = p.name;
      return option;
    }),
  );
  els.provider.value = known ? head : (state.info.providers.find((p) => p.ready)?.name ?? "");
  els.modelInput.value = known ? rest.join("/") : settings.model;
  els.system.value = settings.system ?? "";
  els.temperature.value = settings.temperature ?? "";
  els.problem.hidden = !problem;
  els.problem.textContent = problem ?? "";
  loadModelOptions();
  els.dialog.showModal();
}

async function loadModelOptions() {
  const provider = els.provider.value;
  els.modelOptions.replaceChildren();
  if (!provider) return;
  try {
    const ids = await getJSON(`/api/models/${encodeURIComponent(provider)}`);
    if (els.provider.value !== provider) return;
    els.modelOptions.replaceChildren(...ids.map((id) => { const o = el("option"); o.value = id; return o; }));
  } catch {
    // free text still works
  }
}

async function saveSettings(event) {
  event.preventDefault();
  const typed = els.modelInput.value.trim();
  const model = typed.includes("/") || !els.provider.value ? typed : `${els.provider.value}/${typed}`;
  const temperature = els.temperature.value === "" ? null : Number(els.temperature.value);
  const next = { model, system: els.system.value.trim() || null, temperature, web: currentSettings().web };
  if (!model) {
    els.problem.hidden = false;
    els.problem.textContent = "Enter a model.";
    return;
  }
  try {
    await changeSettings(next);
    els.dialog.close();
  } catch (error) {
    els.problem.hidden = false;
    els.problem.textContent = error.message;
  }
}

// -- input ------------------------------------------------------------------------------

function autosize() {
  els.prompt.style.height = "auto";
  els.prompt.style.height = `${Math.min(els.prompt.scrollHeight, 200)}px`;
}

els.prompt.addEventListener("input", autosize);
els.prompt.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.isComposing) return;
  if (event.shiftKey || event.altKey) return; // a newline
  event.preventDefault();
  send();
});
els.composer.addEventListener("submit", (event) => { event.preventDefault(); send(); });
els.stop.addEventListener("click", stop);
els.newChat.addEventListener("click", newChat);
els.web.addEventListener("click", toggleWeb);
els.thinking.addEventListener("click", toggleAllThinking);
els.settingsOpen.addEventListener("click", () => openSettings());
els.model.addEventListener("click", () => openSettings());
els.provider.addEventListener("change", loadModelOptions);
els.form.addEventListener("submit", saveSettings);
els.cancel.addEventListener("click", () => els.dialog.close());
document.addEventListener("keydown", (event) => {
  if (event.altKey && event.code === "KeyT") {
    event.preventDefault();
    toggleAllThinking();
  }
});

// -- start --------------------------------------------------------------------------------

function heartbeat() {
  request("POST", "/api/heartbeat", { owner: OWNER }).catch(() => {});
}

async function start() {
  state.info = await getJSON("/api/state");
  state.draft.model = state.info.default_model ?? "";
  renderMessages([]);
  updateHeader();
  await refreshChats();
  heartbeat();
  setInterval(heartbeat, 5000);
  setInterval(() => {
    if (state.abort || state.chats.some((chat) => chat.running)) refreshChats();
  }, 3000);
  els.prompt.focus();
}

start().catch((error) => flash(`Could not reach the server: ${error.message}`));
