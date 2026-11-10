"use strict";

// Every piece of text from the server goes in through textContent or text nodes, never as HTML,
// because answers and excerpts are model output and document text.

const $ = (id) => document.getElementById(id);
const CITATION = /(\[\d+(?:\s*[,;–-]\s*\d+)*\])/;

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    const detail = body && typeof body.detail === "string" ? body.detail : `Помилка ${response.status}.`;
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return body;
}

function showLogin() {
  $("whoami").hidden = true;
  $("ask-view").hidden = true;
  $("login-view").hidden = false;
  $("login").focus();
}

function showUser(me) {
  $("login-view").hidden = true;
  $("ask-view").hidden = false;
  $("whoami").hidden = false;
  $("user-name").textContent = me.title ? `${me.name}, ${me.title}` : me.name;
  $("user-roles").textContent = `ролі: ${me.roles.join(", ")}`;
  const select = $("model");
  select.replaceChildren();
  for (const name of me.models) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    option.selected = name === me.default_model;
    select.append(option);
  }
  loadDocuments();
  $("question").focus();
}

async function loadDocuments() {
  const list = $("doc-list");
  list.replaceChildren();
  try {
    for (const doc of await api("/api/documents")) {
      const item = document.createElement("li");
      item.textContent = `${doc.title} (${doc.version})`;
      list.append(item);
    }
  } catch (error) {
    const item = document.createElement("li");
    item.textContent = error.message;
    list.append(item);
  }
}

function renderAnswer(data) {
  const text = $("answer-text");
  text.replaceChildren();
  text.classList.toggle("refused", data.refused);
  for (const part of data.answer.split(CITATION)) {
    if (CITATION.test(part)) {
      const sup = document.createElement("sup");
      for (const number of part.match(/\d+/g)) {
        const link = document.createElement("a");
        link.href = `#source-${number}`;
        link.textContent = `[${number}]`;
        sup.append(link);
      }
      text.append(sup);
    } else {
      text.append(document.createTextNode(part));
    }
  }

  const sources = $("sources");
  sources.replaceChildren();
  for (const citation of data.citations) {
    const item = document.createElement("li");
    item.id = `source-${citation.number}`;
    item.value = citation.number;
    const label = document.createElement("strong");
    label.textContent = citation.label;
    item.append(label);
    if (citation.url) {
      const link = document.createElement("a");
      link.href = citation.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = "першоджерело";
      item.append(" · ", link);
    }
    const excerpt = document.createElement("blockquote");
    excerpt.textContent = citation.excerpt;
    item.append(excerpt);
    sources.append(item);
  }
  const seconds = (data.timings_ms.total_ms / 1000).toFixed(1);
  $("meta").textContent = `Модель: ${data.model} · ${seconds} с`;
  $("answer").hidden = false;
}

$("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("login-error").textContent = "";
  try {
    const me = await api("/api/login", {
      method: "POST",
      body: JSON.stringify({ login: $("login").value.trim(), password: $("password").value }),
    });
    $("password").value = "";
    showUser(me);
  } catch (error) {
    $("login-error").textContent = error.message;
  }
});

$("logout").addEventListener("click", async () => {
  await api("/api/logout", { method: "POST" }).catch(() => null);
  $("answer").hidden = true;
  showLogin();
});

$("ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("ask-button");
  button.disabled = true;
  $("status").textContent = "Шукаю в документах…";
  $("answer").hidden = true;
  try {
    const data = await api("/api/ask", {
      method: "POST",
      body: JSON.stringify({ question: $("question").value, model: $("model").value }),
    });
    $("status").textContent = "";
    renderAnswer(data);
  } catch (error) {
    if (error.status === 401) {
      showLogin();
    }
    $("status").textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

api("/api/me").then(showUser).catch(showLogin);
