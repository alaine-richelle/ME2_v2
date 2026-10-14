"use strict";
const $ = id => document.getElementById(id);
let current = null, mappings = {}, actions = {}, mappingReady = false, lastEvent = -1;
let disconnected = false;
const descriptions = {
  STOPPED: ["Stopped", "Start listening when you’re ready."],
  STARTING: ["Loading models", "Checking model contracts and microphone settings."],
  LISTENING_FOR_WAKE: ["Listening for wake", "Say “a-teh-ko”, then your command."],
  CAPTURING_COMMAND: ["Capturing command", "Listening for speech and its endpoint."],
  DECODING: ["Decoding", "Turning captured speech into a command."],
  ACCEPTED_OR_REJECTED: ["Decision ready", "Checking acceptance and command mapping."],
  COOLDOWN: ["Cooldown", "Suppressing duplicate triggers before listening again."],
  STOPPING: ["Stopping", "Finishing active inference and releasing the microphone."],
  ERROR: ["Needs attention", "Read the error above, correct the setup, then retry."],
};
function text(id, value) { $(id).textContent = value; }
function showError(message) { text("error", message || ""); $("error").hidden = !message; }
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {cache: "no-store"} : {
    method: "POST", headers: {"Content-Type": "application/json", "X-ME2-Token": current?.token || ""},
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}
async function command(path, body = {}) {
  try { await api(path, body); showError(null); }
  catch (error) { showError(error.message); }
}
function renderMappings() {
  const rows = $("mapping-rows");
  rows.replaceChildren();
  for (const [intent, mapping] of Object.entries(mappings)) {
    const row = document.createElement("tr");
    const enabledCell = document.createElement("td");
    const enabled = document.createElement("input");
    enabled.type = "checkbox"; enabled.checked = mapping.enabled;
    enabled.setAttribute("aria-label", `Enable ${intent}`);
    enabled.addEventListener("change", () => { mapping.enabled = enabled.checked; text("mapping-feedback", "Unsaved changes"); });
    enabledCell.append(enabled);
    const name = document.createElement("td"); name.textContent = intent;
    const actionCell = document.createElement("td");
    actionCell.textContent = actions[mapping.action] || "Unavailable";
    const support = document.createElement("td");
    support.className = "support"; support.dataset.intent = intent;
    row.append(enabledCell, name, actionCell, support); rows.append(row);
  }
  const demo = $("demo-intent");
  for (const intent of Object.keys(mappings)) {
    const option = document.createElement("option");
    option.value = intent; option.textContent = intent.replaceAll("_", " "); demo.append(option);
  }
  demo.value = "light_on";
}
function render(data) {
  current = data;
  if (disconnected) { showError(null); disconnected = false; }
  $("save-mappings").disabled = !mappingReady;
  text("connection", "Connected · local service");
  text("mode", data.mode === "demo" ? "SYNTHETIC DEMO" : "LIVE MICROPHONE");
  text("notice", data.notice);
  const [label, detail] = descriptions[data.state] || [data.state, ""];
  text("state", label); text("state-detail", detail);
  text("live-label", data.mode === "demo" ? "SIMULATED AUDIO" : (data.running ? "MIC SESSION" : "MIC OFF"));
  document.querySelectorAll("[data-state]").forEach(el => el.classList.toggle("active", el.dataset.state === data.state));
  $("start").disabled = data.running; $("stop").disabled = !data.running || data.stopping;
  $("level").value = data.rms; text("rms", `${data.rms.toFixed(3)} RMS`);
  $("demo-panel").hidden = data.mode !== "demo";
  const ready = data.state === "LISTENING_FOR_WAKE" && !data.stopping;
  $("demo-send").disabled = !ready; $("demo-reject").disabled = !ready;
  if (data.error) showError(data.error);
  text("wake-count", data.counts.wake);
  text("result-counts", `${data.counts.accepted} / ${data.counts.rejected}`);
  text("memory", `${data.metrics.rss_mb} MB`);
  text("system-detail", `${data.metrics.cpu_percent.toFixed(1)}% CPU · process, one-core scale`);
  text("input-device", data.mode === "demo" ? "Synthetic audio · microphone unused" : (data.device ?? "System default"));
  text("temperature", data.metrics.temperature_c === null ? "Unavailable on this host" : `${data.metrics.temperature_c.toFixed(1)} °C`);
  if (data.wake) {
    text("wake-score", data.wake.score.toFixed(3));
    text("wake-detail", `${data.wake.label} · ${data.mode === "demo" ? "simulated" : `${data.wake.inference_ms.toFixed(1)} ms`}`);
  } else { text("wake-score", "—"); text("wake-detail", "Waiting for a window"); }
  const result = data.last_result;
  if (result) {
    text("result-badge", result.accepted ? "ACCEPTED" : "REJECTED");
    $("result-badge").className = result.accepted ? "badge" : "badge rejected";
    text("transcript", result.decoded_text || (result.accepted ? "Intent classifier result" : "No command captured"));
    text("intent", result.actionable_intent || "—");
    text("slots", Object.keys(result.slots).length ? JSON.stringify(result.slots) : "None returned");
  text("action", data.last_action ? `${data.last_action_status || "result"} · ${data.last_action}` : "—");
    text("rejection", result.rejection_reason || "—");
    text("decoder", JSON.stringify({
      normalized_log_score: result.normalized_log_score, candidate_posterior: result.candidate_posterior,
      margin: result.margin, stage_timings_ms: result.stage_timings_ms,
    }, null, 2));
  } else {
    text("result-badge", "WAITING"); $("result-badge").className = "badge neutral";
    text("transcript", "Your next command will appear here.");
    for (const id of ["intent", "slots", "action", "rejection"]) text(id, "—");
    text("decoder", "No result yet.");
  }
  $("lamp").classList.toggle("on", data.devices.light);
  text("light-state", data.devices.light ? "Light on" : "Light off");
  text("player-state", `${data.devices.player} · track ${data.devices.track}`);
  text("volume-state", `Volume ${data.devices.volume}%`);
  text("brightness-state", `Brightness ${data.devices.brightness}%`);
  text("color-state", `Color ${data.devices.color || "—"}`);
  text("thermostat-state", `Demo thermostat ${data.devices.thermostat_c}°`);
  text("time-state", data.devices.time
    ? `Local time ${new Date(data.devices.time).toLocaleTimeString()}`
    : "Local time —");
  const weather = data.devices.weather || {};
  const weatherStatus = {
    idle: ["NOT REQUESTED", "badge neutral"],
    loading: ["LOADING", "badge neutral"],
    cached: ["CACHED", "badge neutral"],
    ready: ["CURRENT", "badge"],
    stale: ["CACHED · STALE", "badge rejected"],
    error: ["UNAVAILABLE", "badge rejected"],
  }[weather.status] || ["UNKNOWN", "badge neutral"];
  text("weather-status", weatherStatus[0]);
  $("weather-status").className = weatherStatus[1];
  text("weather-location", `Configured city: ${weather.city || "not configured"}`);
  const current = weather.current;
  text("weather-current", current
    ? `${current.condition} · ${current.temperature}${current.temperature_unit} · feels like ${current.apparent_temperature}${current.temperature_unit} · humidity ${current.relative_humidity_percent}%`
    : (weather.status === "loading" ? "Fetching a current observation…" : "No current observation available."));
  text("weather-observed", current
    ? `Observed ${current.observation_time}${current.timezone ? ` · ${current.timezone}` : ""}${weather.cached_at ? ` · fetched ${new Date(weather.cached_at).toLocaleTimeString()}` : ""}`
    : (weather.error || "No observation cached."));
  text("call-state", data.devices.call_demo?.message || "No demo call action.");
  text("message-state", data.devices.message_demo?.message || "No demo message prepared.");
  const reminderRows = data.devices.reminders || [];
  text("reminder-state", reminderRows.length
    ? `Demo reminders · ${reminderRows.map(item => item.text).join(", ")}${data.devices.reminders_viewed_at ? " · list viewed" : ""}`
    : "No demo reminders yet.");
  const renderActionList = (id, rows, empty, formatter) => {
    const list = $(id); list.replaceChildren();
    const items = rows.length ? rows : [empty];
    for (const item of items) {
      const li = document.createElement("li");
      if (typeof item === "string") li.textContent = item;
      else li.textContent = formatter(item);
      list.append(li);
    }
  };
  renderActionList("timer-list", data.devices.timers || [], "No demo timers.",
    timer => `${timer.label} · ${Math.max(0, Math.ceil(timer.ends_at - Date.now() / 1000))} sec remaining`);
  renderActionList("alarm-list", data.devices.alarms || [], "No demo alarms.",
    alarm => `${alarm.label} · ${alarm.time}`);
  document.querySelectorAll(".support").forEach(el => {
    el.textContent = data.mode === "demo" ? "Demo" : data.supported_intents.includes(el.dataset.intent) ? "In loaded model" : "Not loaded / unavailable";
  });
  text("models", data.mode === "demo" ? "Synthetic demo. No model weights loaded." : Object.keys(data.models).length ? JSON.stringify(data.models, null, 2) : "Models load when microphone listening starts.");
  text("grammar", data.grammar?.length ? JSON.stringify(data.grammar, null, 2) : "No live grammar loaded. Legacy grammar entries may have no slots.");
  const latest = data.events[0]?.id ?? 0;
  if (latest !== lastEvent) {
    lastEvent = latest;
    const events = $("events"); events.replaceChildren();
    if (!data.events.length) { const li = document.createElement("li"); li.textContent = "No events yet."; events.append(li); }
    for (const event of data.events) {
      const li = document.createElement("li"), time = document.createElement("time"), title = document.createElement("span"), detail = document.createElement("span");
      time.dateTime = event.at; time.textContent = new Date(event.at).toLocaleTimeString();
      title.textContent = event.message.replaceAll("_", " ");
      detail.className = "event-description"; detail.textContent = [event.intent, event.action].filter(Boolean).join(" · ") || event.state;
      li.append(time, title, detail); events.append(li);
    }
  }
}
async function poll() {
  try {
    render(await api("/api/status"));
    if (!mappingReady) {
      const payload = await api("/api/mappings");
      mappings = payload.mappings; actions = payload.actions; renderMappings(); mappingReady = true;
      text("mapping-feedback", "Saved mappings are applied by the host.");
    }
  } catch (error) {
    disconnected = true;
    text("connection", "Disconnected · retrying");
    for (const id of ["start", "stop", "demo-send", "demo-reject", "save-mappings"]) $(id).disabled = true;
    showError(`Cannot reach the dashboard service: ${error.message}`);
  } finally { setTimeout(poll, 400); }
}
$("start").addEventListener("click", () => command("/api/start"));
$("stop").addEventListener("click", () => command("/api/stop"));
$("demo-send").addEventListener("click", () => command("/api/demo", {intent: $("demo-intent").value, accepted: true}));
$("demo-reject").addEventListener("click", () => command("/api/demo", {intent: $("demo-intent").value, accepted: false}));
$("save-mappings").addEventListener("click", async () => {
  try { await api("/api/mappings", mappings); text("mapping-feedback", "Enable/disable settings saved. Canonical actions are fixed."); showError(null); }
  catch (error) { showError(error.message); }
});
poll();
