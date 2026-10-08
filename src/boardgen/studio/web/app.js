import { renderWorkflow, updateReadiness } from "./workflow.js";
import { editConfiguration } from "./configuration.js";
import {
  state,
  api,
  applyProject,
  refresh,
  loadJobs,
  save,
  notice,
  esc,
  setPath,
  getPath,
  kindNames,
  statusNames,
} from "./api.js";
import { wireHelp, wirePages, pagination } from "./controls.js";
import { showArtifact } from "./results.js";
import * as project from "./project.js";
import * as prepare from "./prepare.js";
import * as dataset from "./dataset.js";
const pages = { project, prepare, dataset };
const titles = {
  project: ["Проект", "Источники и окружение для работы."],
  prepare: ["Подготовка фигур", "Видео, маски, 3D-форма и текстуры."],
  dataset: ["Датасет", "Сцены с автоматической разметкой и разделение данных."],
};
let saveTimer,
  activeLog = null,
  lastStatuses = "",
  polling = false,
  dirty = false;
function savingLabel(text, error = false) {
  const el = document.querySelector("#save-state");
  if (el) {
    el.textContent = text;
    el.classList.toggle("error-text", error);
  }
}
export function onChange(redraw = false) {
  clearTimeout(saveTimer);
  dirty = true;
  savingLabel("Есть изменения");
  saveTimer = setTimeout(async () => {
    saveTimer = null;
    savingLabel("Сохранение…");
    try {
      await save();
      dirty = false;
      savingLabel("Сохранено");
      await updateReadiness();
    } catch (e) {
      savingLabel("Ошибка настроек", true);
      notice(e.message, true);
    }
  }, 350);
  if (redraw) {
    render();
    savingLabel("Есть изменения");
  }
}
async function flush() {
  clearTimeout(saveTimer);
  await save();
  dirty = false;
  savingLabel("Сохранено");
}
function navigate(page) {
  state.page = page;
  sessionStorage.setItem("page", page);
  render();
  window.scrollTo({ top: 0 });
}
async function run(kind, parameters = {}) {
  try {
    await flush();
    await api("/jobs", {
      method: "POST",
      body: JSON.stringify({ kind, parameters }),
    });
    state.listPages.jobs = 0;
    await loadJobs();
    renderJobs();
    notice("Задание добавлено в очередь");
    document
      .querySelector("#jobs")
      ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (e) {
    notice(e.message, true);
  }
}
function bind() {
  document.querySelectorAll("[data-bind]").forEach(
    (control) =>
      (control.onchange = () => {
        if (!control.checkValidity()) {
          control.reportValidity();
          return;
        }
        const path = control.dataset.bind;
        const old = getPath(
          path.startsWith("prep.")
            ? state.project.prepare.defaults
            : state.project,
          path.replace(/^prep\./, ""),
        );
        let value =
          control.type === "checkbox"
            ? control.checked
            : control.type === "number"
              ? control.value === "" && path === "evaluation.imgsz"
                ? null
                : Number(control.value)
              : control.value;
        if (typeof old === "number") value = Number(value);
        if (path === "scenes.reference_class" && value === "") value = null;
        if (path.startsWith("prep.")) {
          const actual = path.slice(5);
          const target =
            state.scope === "all"
              ? state.project.prepare.defaults
              : (state.project.prepare.overrides[state.scope] ??= {});
          setPath(target, actual, value);
          if (
            state.scope !== "all" &&
            getPath(state.project.prepare.defaults, actual) === value
          ) {
            const keys = actual.split(".");
            delete target[keys[0]][keys[1]];
            if (!Object.keys(target[keys[0]]).length) delete target[keys[0]];
            if (!Object.keys(target).length)
              delete state.project.prepare.overrides[state.scope];
          }
          if (
            (actual === "shape.method" && value === "mv") ||
            (actual === "shape.extraction" && value === "depth_fusion")
          ) {
            setPath(target, "appearance.method", "photo");
            if (actual === "shape.method")
              setPath(target, "shape.extraction", "native");
          }
          if (
            actual === "matte.method" ||
            actual === "matte.input_size" ||
            actual === "shape.candidate_frames" ||
            actual === "shape.frame_size"
          )
            for (const figure of state.scope === "all"
              ? state.project.figures
              : state.project.figures.filter((f) => f.id === state.scope)) {
              figure.reviewed_views = {};
              figure.reviewed_frames_id = null;
            }
        } else setPath(state.project, path, value);
        onChange(control.tagName === "SELECT" || control.type === "checkbox");
      }),
  );
  document.querySelectorAll("[data-figure-field]").forEach(
    (control) =>
      (control[control.type === "number" ? "onchange" : "oninput"] = () => {
        const f = state.project.figures.find(
          (f) => f.id === control.dataset.figure,
        );
        const path = control.dataset.figureField;
        const value =
          control.type === "number"
            ? control.value === ""
              ? null
              : Number(control.value)
            : control.value;
        setPath(f, path, value);
        if (path.startsWith("clip.")) {
          f.reviewed_views = {};
          f.reviewed_frames_id = null;
        }
        if (path === "name") {
          const option = document.querySelector(
            `#prepare-scope option[value="${f.id}"]`,
          );
          if (option) option.textContent = f.name + " отдельно";
        }
        onChange();
      }),
  );
  document.querySelectorAll("[data-height-figure]").forEach(
    (control) =>
      (control.onchange = () => {
        if (control.value === "")
          delete state.project.scenes.height_overrides[
            control.dataset.heightFigure
          ];
        else
          state.project.scenes.height_overrides[control.dataset.heightFigure] =
            Number(control.value);
        onChange();
      }),
  );
  wireHelp();
}
function render() {
  prepare.dispose();
  if (!pages[state.page]) state.page = "project";
  const [title, subtitle] = titles[state.page];
  document.querySelector("#app").innerHTML =
    `<div class="shell"><aside class="sidebar"><div class="brand">Boardgen <em>Studio</em></div><div class="project-name">${esc(state.project.name)}</div><nav class="nav" aria-label="Этапы проекта">${Object.entries(
      titles,
    )
      .map(
        ([key, [label]], i) =>
          `<button class="${state.page === key ? "active" : ""}" data-page="${key}" aria-current="${state.page === key ? "page" : "false"}"><span class="step-number">${i + 1}</span>${label}</button>`,
      )
      .join(
        "",
      )}</nav><div class="sidebar-footer">Локальный проект<br><span id="save-state">Сохранено</span></div></aside><main class="main"><header class="page-header"><div><p class="eyebrow">${esc(state.project.name)}</p><h1>${title}</h1><p class="muted">${subtitle}</p></div><span class="badge">${state.project.figures.length} ${state.project.figures.length % 100 >= 11 && state.project.figures.length % 100 <= 14 ? "фигур" : state.project.figures.length % 10 === 1 ? "фигура" : [2, 3, 4].includes(state.project.figures.length % 10) ? "фигуры" : "фигур"}</span></header>${renderWorkflow()}${pages[state.page].render()}<section class="jobs" id="jobs" aria-label="Задания проекта"></section></main></div>`;
  document.querySelectorAll("[data-page]").forEach(
    (b) =>
      (b.onclick = () => {
        if (b.dataset.prepareStage) state.stage = b.dataset.prepareStage;
        navigate(b.dataset.page);
      }),
  );
  bind();
  const guardedRedraw = () => render();
  if (state.page === "project")
    project.wire(navigate, guardedRedraw, run, flush);
  else
    pages[state.page].wire(guardedRedraw, run, showArtifact, onChange, flush);
  wirePages(document.querySelector(".main"), guardedRedraw);
  document.querySelector("[data-config-project]").onclick = () =>
    editConfiguration(render, flush);
  document
    .querySelector("#accept-near")
    ?.addEventListener("change", updateReadiness);
  updateReadiness();
  renderJobs();
}
window.addEventListener("workflow-change", () => {
  const section = document.querySelector(".workflow");
  if (section) {
    section.outerHTML = renderWorkflow();
    document.querySelector("[data-config-project]").onclick = () =>
      editConfiguration(render, flush);
  }
  updateReadiness();
});
function renderJobs() {
  const root = document.querySelector("#jobs");
  if (!root) return;
  root.innerHTML = state.jobs.length
    ? `<div class="actions spaced"><h2 style="margin:0">Задания</h2><span class="muted small-text">Тяжёлые задачи выполняются по очереди${state.jobHistory.active_count ? ` · активных: ${state.jobHistory.active_count}` : ""}</span></div><div class="job-list">${state.jobs
        .map(
          (j) =>
            `<article class="job"><div class="job-head"><div><strong>${kindNames[j.kind] || j.kind}</strong><div class="job-time">${new Date(j.created).toLocaleString("ru")} · ${j.id.slice(0, 6)}</div></div><div class="actions"><span class="badge ${j.status === "failed" ? "error" : j.status === "running" ? "warning" : ""}">${statusNames[j.status] || j.status}</span>${["queued", "running"].includes(j.status) ? `<button class="small danger" data-cancel="${j.id}">Отменить</button>` : ""}<button class="small" data-log="${j.id}">Журнал</button>${["failed", "interrupted", "cancelled"].includes(j.status) ? `<button class="small" data-retry="${j.id}">Повторить</button>` : ""}${j.resume_available ? `<button class="small" data-job-resume="${j.id}">Продолжить обучение</button>` : ""}</div></div>${j.error ? `<p class="error-text">${esc(j.error)}</p>` : ""}${j.result?.artifacts?.length ? `<div class="actions" style="margin-top:10px">${j.result.artifacts.map((id) => `<button class="small" data-result="${id}">Результат ${id.slice(0, 6)}</button>`).join("")}</div>` : ""}${activeLog === j.id ? `<pre class="logs" id="log-${j.id}">Загрузка журнала…</pre>` : ""}</article>`,
        )
        .join("")}</div>${pagination("jobs", state.jobHistory.total, 5)}`
    : "";
  root.querySelectorAll("[data-cancel]").forEach(
    (b) =>
      (b.onclick = async () => {
        await api(`/jobs/${b.dataset.cancel}/cancel`, { method: "POST" });
        await poll();
      }),
  );
  root.querySelectorAll("[data-log]").forEach(
    (b) =>
      (b.onclick = async () => {
        activeLog = activeLog === b.dataset.log ? null : b.dataset.log;
        renderJobs();
        await loadLog();
      }),
  );
  root.querySelectorAll("[data-retry]").forEach(
    (b) =>
      (b.onclick = () => {
        const job = state.jobs.find((j) => j.id === b.dataset.retry);
        run(job.kind, job.parameters);
      }),
  );
  root
    .querySelectorAll("[data-result]")
    .forEach((b) => (b.onclick = () => showArtifact(b.dataset.result)));
  root
    .querySelectorAll("[data-job-resume]")
    .forEach(
      (b) =>
        (b.onclick = () =>
          run("train", { resume_job_id: b.dataset.jobResume })),
    );
  wirePages(root, async () => {
    activeLog = null;
    await loadJobs();
    renderJobs();
  });
  loadLog();
}
async function loadLog() {
  if (!activeLog || !document.querySelector(`#log-${activeLog}`)) return;
  try {
    const data = await api(`/jobs/${activeLog}/log`);
    const el = document.querySelector(`#log-${activeLog}`);
    if (el) {
      const bottom = el.scrollHeight - el.scrollTop - el.clientHeight < 50;
      el.textContent = data.text || "Задание ещё не запущено.";
      if (bottom) el.scrollTop = el.scrollHeight;
    }
  } catch {}
}
async function poll() {
  if (polling || document.hidden) return;
  polling = true;
  try {
    const previous = new Map(state.jobs.map((j) => [j.id, j.status]));
    const previousResume = state.jobHistory.latest_resume?.id;
    const history = await loadJobs();
    const jobs = history.items;
    const statuses = JSON.stringify([
      jobs.map((j) => [j.id, j.status, j.updated]),
      history.total,
      history.active_count,
      history.latest_resume?.id,
    ]);
    if (statuses !== lastStatuses) {
      lastStatuses = statuses;
      const oldArtifacts = state.artifacts.map((a) => a.id).join();
      state.artifacts = await api("/artifacts");
      for (const job of jobs)
        if (job.status === "success" && previous.get(job.id) === "running")
          notice(`${kindNames[job.kind]}: готово`);
      const completed = jobs.some(
        (j) => j.status === "success" && previous.get(j.id) === "running",
      );
      if (
        (completed ||
          oldArtifacts !== state.artifacts.map((a) => a.id).join() ||
          previousResume !== history.latest_resume?.id) &&
        !dirty &&
        !state.saving &&
        !["INPUT", "SELECT", "TEXTAREA"].includes(
          document.activeElement?.tagName,
        )
      )
        render();
      else renderJobs();
    } else await loadLog();
  } catch {
  } finally {
    polling = false;
  }
}
window.addEventListener("unhandledrejection", (event) => {
  notice(event.reason?.message || String(event.reason), true);
  event.preventDefault();
});
try {
  applyProject(await api("/project"));
  await refresh();
  render();
  setInterval(poll, 2000);
} catch (e) {
  document.querySelector("#app").innerHTML =
    `<p class="loading error-text">Не удалось открыть проект: ${esc(e.message)}</p>`;
}
