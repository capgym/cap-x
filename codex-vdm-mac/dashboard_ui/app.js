const el = (id) => document.getElementById(id);
const picker = el("run-select");
const stream = el("stream");
const eventList = el("events");
const empty = el("empty");
const video = el("camera-video");
const cameraImage = el("camera-image");
const combinedButton = el("combined");
const latestButton = el("latest");
const runScopes = Array.from(document.querySelectorAll("[data-run-group]"));
const models = { qwen: "Qwen", codex: "Codex", sim: "仿真" };
const statuses = { missing: "未保存", reconstructed: "已还原", partial: "部分记录", network_error: "连接失败", stream_error: "输出中断" };
const roles = { system: "系统", user: "输入", assistant: "模型", developer: "开发者", tool: "工具" };
const filters = Array.from(document.querySelectorAll("[data-filter]"));
let requestedRun = new URLSearchParams(window.location.search).get("run");

let runs = new Map();
let activeRun = "";
let pickerGroup = null;
const groupSelections = new Map();
const currentOrder = { running: 0, finalizing: 1, pending: 2 };
let generation = 0;
let offset = 0;
let cursor = null;
let activeFilter = "all";
let lastFrame = null;
let combinedVideo = null;
let replayPoster = null;
let selectedMedia = null;
let cameraPinned = false;
let liveMedia = null;
const exchanges = new Map();
let currentRound = 0;
let currentGroup = null;

function node(tag, className, contents) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (contents !== undefined) element.textContent = contents;
  return element;
}

function setConnection(label, status) {
  el("connection-label").textContent = label;
  el("connection").dataset.status = status;
}

function mediaUrl(runId, path) {
  return "/media/" + encodeURIComponent(runId) + "/" + path.split("/").map(encodeURIComponent).join("/");
}

function showMedia(src, isVideo, caption) {
  if (selectedMedia !== src) {
    selectedMedia = src;
    if (!video.hidden) video.pause();
    video.hidden = !isVideo;
    cameraImage.hidden = isVideo;
    el("camera-empty").hidden = true;
    if (isVideo) {
      video.poster = replayPoster ?? "";
      video.src = src;
      cameraImage.removeAttribute("src");
    } else {
      cameraImage.src = src;
      if (video.hasAttribute("src")) {
        video.removeAttribute("src");
        video.load();
      }
    }
  }
  el("camera-caption").textContent = caption;
  combinedButton.hidden = !combinedVideo || combinedVideo === src;
  latestButton.hidden = !liveMedia || liveMedia.src === src;
}

function resetCamera() {
  lastFrame = combinedVideo = selectedMedia = replayPoster = liveMedia = null;
  cameraPinned = false;
  video.pause();
  video.hidden = cameraImage.hidden = combinedButton.hidden = latestButton.hidden = true;
  video.removeAttribute("src");
  video.removeAttribute("poster");
  video.load();
  cameraImage.removeAttribute("src");
  el("camera-empty").hidden = false;
  el("camera-caption").textContent = "相机视角";
  el("camera-mode").textContent = "等待数据";
  delete el("camera-mode").dataset.live;
}

function selectMedia(url, isVideo, label) {
  cameraPinned = true;
  showMedia(url, isVideo, label);
  if (window.matchMedia("(max-width: 720px)").matches) el("camera-title").scrollIntoView({ block: "nearest" });
}

function imageButton(url, label) {
  const button = node("button", "image-button");
  button.type = "button";
  button.setAttribute("aria-label", label);
  const image = node("img", "asset-image");
  image.alt = label;
  image.loading = "lazy";
  image.src = url;
  button.append(image);
  button.addEventListener("click", () => selectMedia(url, false, label));
  return button;
}

function appendMessageContent(parent, content) {
  if (typeof content === "string") {
    parent.append(node("div", "payload-text", content));
    return;
  }
  if (!Array.isArray(content)) {
    if (content != null) parent.append(node("div", "payload-text", JSON.stringify(content, null, 2)));
    return;
  }
  let imageNumber = 0;
  for (const part of content) {
    if (typeof part?.text === "string") {
      parent.append(node("div", "payload-text", part.text));
    } else {
      const url = part?.image_url?.url ?? part?.image_url;
      if (typeof url === "string" && /^data:image\/(?:jpe?g|png|webp|gif);base64,/i.test(url)) {
        parent.append(imageButton(url, "请求图片 " + ++imageNumber));
      } else {
        parent.append(node("div", "payload-text", JSON.stringify(part, null, 2)));
      }
    }
  }
}

function messageText(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.map((part) => part?.text ?? "").filter(Boolean).join("\n");
}

function promptPreview(body) {
  const messages = body?.messages ?? body?.input;
  const user = Array.isArray(messages) && messages.findLast((message) => message?.role === "user");
  const firstParagraph = user && messageText(user.content).trim().split(/\n\s*\n/, 1)[0];
  if (!firstParagraph) return "";
  return firstParagraph.length > 240 ? firstParagraph.slice(0, 239) + "…" : firstParagraph;
}

function requestPurpose(body, model) {
  const messages = body?.messages;
  if (!Array.isArray(messages)) return "";
  const system = messages.find((message) => message?.role === "system");
  const text = messageText(system?.content);
  if (model === "codex") {
    if (promptPreview(body).startsWith("What color fills this image?")) return "连通检查";
    if (text.includes("describes the initial state")) return "初始观察";
    if (text.includes("describes the difference")) return "视觉差分";
  }
  if (model === "qwen" && promptPreview(body).startsWith("What color fills this image?")) return "连通检查";
  return "";
}

function requestDetails(body) {
  const messages = body?.messages ?? body?.input;
  if (!Array.isArray(messages)) return null;
  const details = node("details");
  details.append(node("summary", "", "查看完整提示与图片"));
  details.addEventListener("toggle", () => {
    if (!details.open || details.children.length > 1) return;
    for (const message of messages) {
      const row = node("div", "payload-row");
      row.append(node("div", "role-label", roles[message?.role] ?? String(message?.role ?? "Message")));
      appendMessageContent(row, message?.content ?? message);
      details.append(row);
    }
  });
  return details;
}

function replyContent(body) {
  if (typeof body === "string") return body;
  const choices = body?.choices;
  if (Array.isArray(choices) && choices.length) {
    const messages = choices.map((choice) => {
      const message = choice?.message;
      return message?.content ?? choice?.text ?? message ?? "";
    });
    return messages.length === 1 ? messages[0] : messages.map((msg, idx) => "回复 " + (idx + 1) + "\n" + (typeof msg === "string" ? msg : JSON.stringify(msg, null, 2))).join("\n\n");
  }
  return body?.output_text ?? body?.error?.message ?? "";
}

function rawDetails(body) {
  const details = node("details", "raw");
  details.append(node("summary", "", typeof body === "string" ? "原始内容" : "原始 JSON"));
  details.addEventListener("toggle", () => {
    if (!details.open || details.children.length > 1) return;
    details.append(node("pre", "", typeof body === "string" ? body : JSON.stringify(body, null, 2)));
  });
  return details;
}

function codePanel(source, language) {
  const panel = node("div", "code-panel");
  const header = node("div", "code-panel-header");
  const label = language.toLowerCase() === "python" ? "Python" : language || "代码";
  const copy = node("button", "copy-code", "复制");
  copy.type = "button";
  copy.setAttribute("aria-label", "复制代码");
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(source);
      copy.textContent = "已复制";
    } catch {
      copy.textContent = "复制失败";
    }
  });
  header.append(node("span", "", label), copy);
  const pre = node("pre");
  pre.tabIndex = 0;
  pre.append(node("code", "", source));
  panel.append(header, pre);
  return panel;
}

function appendReply(parent, output) {
  if (typeof output !== "string") {
    appendMessageContent(parent, output);
    return;
  }
  // Split fenced blocks without interpreting model text as HTML.
  const fences = /(?:^|\n)[ \t]*```([\w+-]*)[^\S\n]*\n([\s\S]*?)(?:\n[ \t]*```(?=\s|$)|$)/g;
  let end = 0;
  for (const match of output.matchAll(fences)) {
    const prose = output.slice(end, match.index).trim();
    if (prose) parent.append(node("p", "event-text", prose));
    parent.append(codePanel(match[2], match[1]));
    end = match.index + match[0].length;
  }
  const tail = output.slice(end).trim();
  if (tail) parent.append(node("p", "event-text", tail));
}

function reasoningDetails(reasoning) {
  const text = messageText(reasoning);
  if (!text.trim()) return null;
  const details = node("details", "reasoning");
  details.append(node("summary", "", "查看推理"), node("div", "event-text", text));
  return details;
}

function appendMotionNote(parent, text) {
  if (!text) return;
  const cleaned = text.replace(/^\[TRON2[^\]]*\]\s*/, "");
  const dispatch = cleaned.match(/^SERVOJ_STREAM stage=(\S+) (\{.*\})$/);
  const contact = cleaned.match(/^STOPPED_ON_CONTACT (\{.*\})$/);
  const tcp = cleaned.match(/^(.*?) measured TCP error (.*?) actual_pos=.*?reached=(True|False)\s*$/);
  let preview = cleaned;
  if (tcp) {
    preview = `${tcp[1]} · ${tcp[3] === "True" ? "已到位" : "未到位"} · 误差 ${tcp[2]}`;
    parent.dataset.outcome = tcp[3] === "True" ? "ok" : "warning";
  } else if (dispatch || contact) {
    try {
      const data = JSON.parse(dispatch?.[2] ?? contact[1]);
      if (contact) {
        const force = Number.isFinite(data.trigger_force_n) ? ` · ${data.trigger_force_n.toFixed(1)} N` : "";
        preview = `${data.stage ?? "当前动作"} · 碰桌停止${force}`;
        parent.dataset.outcome = "warning";
      } else if (Number.isFinite(data.sent_samples) && Number.isFinite(data.planned_samples)) {
        preview = `${dispatch[1]} · ServoJ 发帧 ${data.sent_samples}/${data.planned_samples}`;
      }
    } catch {
      // Preserve the original message if diagnostics use an older schema.
    }
  } else if (cleaned.startsWith("TABLE_CONTACT") || cleaned.startsWith("WARNING:")) {
    parent.dataset.outcome = "warning";
  }
  parent.append(node("p", "event-text", preview));
  if (preview !== text) {
    const details = node("details", "raw");
    details.append(node("summary", "", "原始日志"), node("pre", "", text));
    parent.append(details);
  }
}

function renderAsset(parent, path, runId, mediaLabel) {
  if (typeof path !== "string") return;
  const url = mediaUrl(runId, path);
  const filename = path.split("/").at(-1);
  if (/\.(jpe?g|png|webp|gif)$/i.test(path)) {
    parent.append(imageButton(url, mediaLabel ?? filename));
  } else if (/\.mp4$/i.test(path)) {
    const button = node("button", "asset-button", "播放片段");
    button.type = "button";
    button.addEventListener("click", () => selectMedia(url, true, mediaLabel ?? filename));
    parent.append(button);
  } else {
    const link = node("a", "", "打开 " + filename);
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener";
    parent.append(link);
  }
}

function renderEvent(event, runId, model, label, round) {
  const request = event.type === "request";
  const reply = event.type === "reply";
  const kind = request ? "request" : reply ? "reply" : "note";
  const card = node("article", `event ${model} event-${kind}`);
  card.dataset.model = model;
  const avatar = node("span", "event-avatar", { qwen: "Q", codex: "C", sim: "仿" }[model]);
  avatar.setAttribute("aria-hidden", "true");
  const body = node("div", "event-body");
  const heading = node("div", "event-head");
  const actor = request ? `发给 ${models[model]}` : models[model];
  heading.append(node("span", "event-model", actor), node("span", "event-title", label));
  if (event.status) {
    const label = statuses[event.status] || (/^\d{3}$/.test(String(event.status)) ? "HTTP " + event.status : String(event.status));
    const badge = node("span", "event-status", label);
    badge.dataset.status = event.status;
    heading.append(badge);
  }
  if (reply && event.body?.choices?.[0]?.finish_reason === "length") {
    const badge = node("span", "event-status", "达到生成上限");
    badge.dataset.status = "partial";
    heading.append(badge);
  }
  body.append(heading);
  const bubble = node("div", "event-bubble");
  const text = ["API request", "API response"].includes(event.text) ? "" : event.text ?? "";
  if (request) {
    const preview = model === "qwen" && round > 1 ? "" : promptPreview(event.body);
    if (preview) bubble.append(node("p", "event-text", preview));
    const details = requestDetails(event.body);
    if (details) bubble.append(details);
    if (text && (!details || text !== preview)) {
      if (text.length > 240) {
        const details = node("details");
        details.append(node("summary", "", event.status === "reconstructed" ? "查看还原请求" : "查看完整消息"),
          node("pre", "payload-text", text));
        bubble.append(details);
      } else {
        bubble.append(node("p", "event-text", text));
      }
    }
  } else if (reply) {
    const message = event.body?.choices?.[0]?.message;
    const reasoning = reasoningDetails(event.reasoning || message?.reasoning || message?.reasoning_content);
    if (reasoning) bubble.append(reasoning);
    if (Array.isArray(event.code_blocks)) {
      if (event.decision) appendReply(bubble, event.decision);
      for (const source of event.code_blocks) bubble.append(codePanel(source, "python"));
    } else {
      const output = replyContent(event.body) || text;
      if (output) appendReply(bubble, output);
      if (text && text !== output) appendReply(bubble, text);
    }
  } else if (text) {
    appendMotionNote(bubble, text);
  }
  if (event.body != null) bubble.append(rawDetails(event.body));
  const assets = node("div", "event-assets");
  const paths = new Set([event.asset, ...(Array.isArray(event.assets) ? event.assets : [])].filter(Boolean));
  for (const path of paths) {
    const mediaLabel = event.model === "sim" && Number.isInteger(event.turn)
      ? `第 ${round} 轮 · ${event.title || (/\.mp4$/i.test(path) ? "仿真片段" : "仿真观察")}`
      : undefined;
    renderAsset(assets, path, runId, mediaLabel);
  }
  bubble.append(assets);
  body.append(bubble);
  card.append(avatar, body);
  return card;
}

function appendQwenDelta(exchange, event, runId) {
  const reasoning = typeof event.reasoning === "string" ? event.reasoning : "";
  const content = typeof event.content === "string" ? event.content : "";
  if (!exchange || exchange.finished || (!reasoning && !content)) return;
  if (!exchange.draft) {
    const card = renderEvent({ type: "reply" }, runId, "qwen", "生成中", exchange.round);
    card.classList.add("event-streaming");
    card.setAttribute("aria-live", "off");
    const thought = node("details", "reasoning stream-reasoning");
    const thoughtBody = node("div", "event-text");
    const thoughtText = document.createTextNode("");
    thoughtBody.append(thoughtText);
    thought.append(node("summary", "", "推理中"), thoughtBody);
    thought.open = true;
    thought.hidden = true;
    const output = node("pre", "stream-output");
    const outputText = document.createTextNode("");
    output.append(outputText);
    output.hidden = true;
    card.querySelector(".event-bubble").prepend(thought, output);
    exchange.draft = { card, thought, thoughtBody, thoughtText, output, outputText };
    exchange.group.append(card);
    exchange.state.textContent = "生成中";
  }
  const draft = exchange.draft;
  if (reasoning) {
    const atEnd = draft.thoughtBody.scrollHeight - draft.thoughtBody.clientHeight - draft.thoughtBody.scrollTop < 24;
    draft.thought.hidden = false;
    draft.thoughtText.appendData(reasoning);
    if (atEnd && el("follow").checked) draft.thoughtBody.scrollTop = draft.thoughtBody.scrollHeight;
  }
  if (content) {
    draft.output.hidden = false;
    draft.outputText.appendData(content);
    draft.thought.querySelector("summary").textContent = "查看推理";
  }
}

function finishReply(exchange, card, failed) {
  exchange.finished = true;
  const draft = exchange.draft;
  if (!draft) return false;
  if (failed) {
    const partial = node("details", "raw");
    partial.append(node("summary", "", "查看已收到的部分输出"));
    if (draft.thoughtText.data) partial.append(node("pre", "", draft.thoughtText.data));
    if (draft.outputText.data) partial.append(node("pre", "", draft.outputText.data));
    card.querySelector(".event-bubble").append(partial);
  }
  draft.card.replaceWith(card);
  exchange.draft = null;
  return true;
}

function stopPendingStreams() {
  for (const exchange of exchanges.values()) {
    const draft = exchange.draft;
    if (!draft || exchange.finished) continue;
    exchange.finished = true;
    exchange.state.textContent = "输出中断";
    draft.card.classList.remove("event-streaming");
    draft.card.querySelector(".event-title").textContent = "部分输出 · 已中断";
    draft.thought.querySelector("summary").textContent = "查看已收到的推理";
  }
}

function appendEvent(event, runId) {
  const model = Object.hasOwn(models, event.model) ? event.model : "sim";
  const matching = event.request_id && exchanges.get(event.request_id);
  if (event.type === "delta") {
    if (model === "qwen") appendQwenDelta(matching, event, runId);
    return;
  }
  const purpose = event.type === "request" ? requestPurpose(event.body, model) : matching?.purpose;
  const annotated = Number.isInteger(event.turn) && event.turn >= 0;
  const startsRound = model === "qwen" && event.type === "request" && purpose !== "连通检查";
  if (annotated) currentRound = event.turn + 1;
  else if (startsRound) currentRound += 1;
  const round = matching?.round ?? currentRound;
  const label = purpose || event.title || ({ request: "输入", reply: model === "qwen" ? "生成结果" : "观察结果",
    frame: "相机画面", note: "执行反馈", finish: "运行结束" })[event.type] || "运行记录";
  const card = renderEvent(event, runId, model, label === "TRON2" ? "执行反馈" : label, round);
  if (event.type === "reply" && matching) {
    const failed = Number(event.status) >= 400 || ["network_error", "stream_error"].includes(event.status);
    matching.state.textContent = statuses[event.status] || (failed ? "请求失败" : "已收到回复");
    if (finishReply(matching, card, failed)) return;
  }
  if (currentGroup?.dataset.round !== String(round)) {
    currentGroup = node("section", "conversation-turn");
    currentGroup.dataset.round = round;
    if (round > 0) currentGroup.append(node("div", "turn-divider", `第 ${round} 轮`));
    eventList.append(currentGroup);
  }
  if (event.type === "request" && event.request_id) {
    const state = node("span", "request-state", "等待回复");
    card.querySelector(".event-head").append(state);
    exchanges.set(event.request_id, { purpose, round, state, group: currentGroup });
  }
  currentGroup.append(card);
}

function updateCount() {
  let count = 0;
  for (const group of eventList.children) {
    let shown = 0;
    for (const card of group.querySelectorAll(".event")) {
      card.hidden = activeFilter !== "all" && card.dataset.model !== activeFilter;
      shown += Number(!card.hidden);
    }
    group.hidden = !shown;
    count += shown;
  }
  el("event-count").textContent = count > 0 ? count + " 条" : "";
  empty.hidden = count > 0;
  empty.textContent = eventList.childElementCount ? "当前筛选没有交互记录" : "等待交互记录";
}

function scrollToLatest() {
  if (el("follow").checked) stream.scrollTop = stream.scrollHeight;
}

function statusLabel(run, state) {
  if (["failed", "error"].includes(state)) return "运行失败";
  if (["aborted", "interrupted"].includes(state)) return "已中断";
  if (run?.mode === "replay") return "历史回放";
  if (state === "pending") return "等待启动";
  if (["finished", "complete", "completed", "done", "succeeded", "success"].includes(state)) return "运行结束";
  if (state === "finalizing") return "正在保存";
  return state === "running" ? "实时运行" : "状态未知";
}

function batchName(runId) {
  return runId.startsWith("tron2:") ? runId.split(":")[1] : null;
}

function shortRunName(runId, name) {
  const batch = batchName(runId);
  const suffix = batch ? ` · ${batch}` : "";
  return suffix && name.endsWith(suffix) ? name.slice(0, -suffix.length) : name;
}

function setRunTitle(runId, name) {
  const title = el("run-title");
  title.textContent = shortRunName(runId, name);
  title.title = name;
}

function updateSummary(summary, values, log) {
  const full = typeof summary === "string" ? summary.trim() : "";
  const metrics = full.match(/Task completed:\s*([\d.]+)\/([\d.]+)\/(\d+)/i);
  const seconds = full.match(/Elapsed time:\s*([\d.]+)/i);
  const trials = full.match(/Total number of trials:\s*(\d+)/i);
  const tags = [];
  if (metrics) tags.push("平均奖励 " + metrics[2], "完成 " + metrics[3] + (trials ? "/" + trials[1] : ""));
  if (seconds) tags.push("耗时 " + Math.round(Number(seconds[1])) + " 秒");
  if (typeof values?.task_completed === "boolean") tags.push(values.task_completed ? "任务完成" : "任务未完成");
  if (Number.isFinite(values?.final_lift_cm)) tags.push("抬升 " + values.final_lift_cm.toFixed(2) + " cm");
  if (Number.isFinite(values?.table_peak_normal_n)) tags.push("碰桌峰值 " + values.table_peak_normal_n.toFixed(1) + " N");
  if (!seconds && Number.isFinite(values?.episode_wall_sec)) tags.push("耗时 " + Math.round(values.episode_wall_sec) + " 秒");
  el("run-summary").textContent = tags.join(" · ");
  el("run-summary").hidden = !tags.length;
  el("run-details").hidden = !full && !log;
  if (el("summary-detail").textContent !== full) el("summary-detail").textContent = full;
  const logLink = el("run-log");
  logLink.hidden = !log;
  if (log) logLink.href = mediaUrl(activeRun, log);
  else logLink.removeAttribute("href");
}

async function json(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error("HTTP " + response.status);
  return response.json();
}

async function pollRun(runId, version) {
  let delay = 1400;
  try {
    const query = new URLSearchParams({ after: offset });
    if (cursor !== null) query.set("cursor", cursor);
    const result = await json("/api/run/" + encodeURIComponent(runId) + "?" + query);
    if (version !== generation) return;
    if (result.reset_cursor) clearTimeline();
    cursor = result.cursor_key ?? null;
    const meta = runs.get(runId);
    setConnection("已连接", "online");
    setRunTitle(runId, result.name || meta?.name || runId);
    const mode = statusLabel(meta, result.state || meta?.state);
    el("run-mode").textContent = mode;
    updateSummary(result.summary, result.metrics, result.log);
    const events = Array.isArray(result.events) ? result.events : [];
    if (!replayPoster) {
      const poster = events.find((event) => event.model === "sim" && typeof event.asset === "string" && /\.(png|jpe?g)$/i.test(event.asset));
      if (poster) replayPoster = mediaUrl(runId, poster.asset);
    }
    if (typeof result.video === "string" && result.video) {
      combinedVideo = mediaUrl(runId, result.video);
      if (!selectedMedia || (!cameraPinned && mode === "运行结束")) showMedia(combinedVideo, true, "合并仿真视频");
      else combinedButton.hidden = combinedVideo === selectedMedia;
    }
    if (result.frame_revision && result.frame_revision !== lastFrame) {
      lastFrame = result.frame_revision;
      liveMedia = { src: "/api/run/" + encodeURIComponent(runId) + "/frame.jpg?rev=" + encodeURIComponent(lastFrame), label: "最新相机画面" };
      if (!cameraPinned && (mode !== "运行结束" || !combinedVideo)) showMedia(liveMedia.src, false, liveMedia.label);
      else latestButton.hidden = false;
    }
    if (!selectedMedia && replayPoster) showMedia(replayPoster, false, "初始相机画面");
    el("camera-mode").textContent = meta?.mode === "replay" ? "回放" : mode === "实时运行" ? (cameraPinned ? "查看附件" : lastFrame ? "实时" : "等待首帧") : mode;
    el("camera-mode").dataset.live = String(mode === "实时运行" && !!lastFrame && !cameraPinned);
    latestButton.textContent = mode === "实时运行" ? "跟随相机" : "查看最后一帧";
    if (events.length) {
      for (const event of events) {
        if (event.type === "frame" && !event.asset && meta?.mode !== "replay") continue;
        appendEvent(event, runId);
      }
    }
    if (result.state && !Object.hasOwn(currentOrder, result.state)) stopPendingStreams();
    offset = Number.isInteger(result.next) ? result.next : offset + events.length;
    if (events.length || !eventList.childElementCount) updateCount();
    if (events.length) scrollToLatest();
    if (meta?.mode === "replay") return;
    delay = mode === "实时运行" ? 350 : 1400;
  } catch (error) {
    if (version !== generation) return;
    setConnection("连接已断开，正在重试", "error");
    if (!eventList.childElementCount) {
      empty.hidden = false;
      empty.textContent = "运行记录暂时无法读取";
    }
  }
  if (version === generation) window.setTimeout(() => pollRun(runId, version), delay);
}

function clearTimeline() {
  exchanges.clear();
  currentRound = 0;
  currentGroup = null;
  eventList.replaceChildren();
  stream.scrollTop = 0;
  resetCamera();
  updateCount();
}

function selectRun(runId) {
  activeRun = runId;
  picker.value = runId;
  const batch = batchName(runId);
  const batchLabel = el("selected-batch");
  batchLabel.textContent = batch ? `批次 ${batch}` : "";
  batchLabel.title = batch || "";
  batchLabel.hidden = !batch;
  offset = 0;
  cursor = null;
  generation += 1;
  const metadata = runs.get(runId);
  if (metadata) groupSelections.set(runGroup(metadata), runId);
  el("follow").checked = metadata ? runGroup(metadata) === "current" : false;
  setRunTitle(runId, metadata?.name || runId || "运行观察台");
  el("run-mode").textContent = metadata ? statusLabel(metadata, metadata.state) : "实验记录";
  el("run-details").open = false;
  updateSummary("");
  clearTimeline();
  if (runId) pollRun(runId, generation);
  else empty.textContent = requestedRun ? "等待这次试次出现" : emptyRunLabel();
}

function runGroup(run) {
  return run.mode !== "replay" && Object.hasOwn(currentOrder, run.state) ? "current" : "history";
}

function groupedRuns(group) {
  const rows = [...runs.values()].filter((run) => runGroup(run) === group);
  if (group === "current") rows.sort((a, b) => currentOrder[a.state] - currentOrder[b.state]);
  return rows;
}

function firstRunInGroup() {
  const rows = groupedRuns(pickerGroup);
  const remembered = groupSelections.get(pickerGroup);
  return rows.find((row) => row.id === remembered)?.id ?? rows[0]?.id ?? "";
}

function emptyRunLabel() {
  return pickerGroup === "current" ? "暂无当前试次" : "暂无历史记录";
}

function renderRunPicker(selected) {
  for (const button of runScopes) {
    const group = button.dataset.runGroup;
    button.disabled = false;
    button.setAttribute("aria-pressed", String(group === pickerGroup));
    button.querySelector(".run-scope-count").textContent = groupedRuns(group).length;
  }
  const rows = groupedRuns(pickerGroup);
  const batches = new Map();
  for (const run of rows) {
    const batch = batchName(run.id) || "CaP-X 原有记录";
    if (!batches.has(batch)) {
      const group = node("optgroup");
      group.label = batch;
      batches.set(batch, group);
    }
    const state = statusLabel(run, run.state);
    const label = shortRunName(run.id, run.name || run.id);
    const option = node("option", "", `${state} · ${label}`);
    option.value = run.id;
    batches.get(batch).append(option);
  }
  const options = [...batches.values()];
  if (!selected) {
    const placeholder = node("option", "", requestedRun ? "等待指定试次…" : emptyRunLabel());
    placeholder.value = "";
    placeholder.disabled = true;
    options.unshift(placeholder);
  }
  picker.replaceChildren(...options);
  picker.disabled = !rows.length;
  picker.value = selected;
}

async function refreshRuns() {
  try {
    const result = await json("/api/runs");
    const newRuns = new Map(result.map((run) => [run.id, run]));
    if (JSON.stringify([...newRuns]) !== JSON.stringify([...runs]) || pickerGroup === null) {
      const initializing = pickerGroup === null;
      runs = newRuns;
      const requested = runs.has(requestedRun) ? requestedRun : null;
      if (requested) requestedRun = null;
      const existing = runs.get(requested ?? activeRun);
      if (existing) {
        pickerGroup = runGroup(existing);
        groupSelections.set(pickerGroup, existing.id);
      }
      if (pickerGroup === null) pickerGroup = groupedRuns("current").length ? "current" : "history";
      const next = existing?.id ?? (requestedRun ? "" : firstRunInGroup());
      renderRunPicker(next);
      if (initializing || next !== activeRun) selectRun(next);
    }
    setConnection("已连接", "online");
  } catch (error) {
    setConnection("连接已断开，正在重试", "error");
    if (!eventList.childElementCount) empty.textContent = "正在等待本地看板服务…";
  } finally {
    window.setTimeout(refreshRuns, 3500);
  }
}

combinedButton.addEventListener("click", () => {
  if (combinedVideo) selectMedia(combinedVideo, true, "合并仿真视频");
});
latestButton.addEventListener("click", () => {
  cameraPinned = el("run-mode").textContent === "运行结束";
  if (liveMedia) showMedia(liveMedia.src, false, liveMedia.label);
});

function pickRun(runId) {
  const url = new URL(window.location.href);
  if (runId) url.searchParams.set("run", runId);
  else url.searchParams.delete("run");
  window.history.replaceState(null, "", url);
  selectRun(runId);
}

picker.addEventListener("change", () => {
  requestedRun = null;
  pickRun(picker.value);
});
for (const button of runScopes) {
  button.addEventListener("click", () => {
    if (button.dataset.runGroup === pickerGroup) return;
    requestedRun = null;
    pickerGroup = button.dataset.runGroup;
    const next = firstRunInGroup();
    renderRunPicker(next);
    pickRun(next);
  });
}
el("follow").addEventListener("change", scrollToLatest);
for (const button of filters) {
  button.addEventListener("click", () => {
    activeFilter = button.dataset.filter;
    for (const filter of filters) filter.setAttribute("aria-pressed", String(filter === button));
    updateCount();
    stream.scrollTop = 0;
    scrollToLatest();
  });
}
refreshRuns();
