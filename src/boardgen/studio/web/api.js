export const state = {
  project: null,
  revision: null,
  root: null,
  token: null,
  artifacts: [],
  jobs: [],
  videos: [],
  page: sessionStorage.getItem("page") || "project",
  stage: "video",
  scope: "all",
  viewTag: "front",
  selectedFramesFigure: null,
  datasetId: null,
  modelId: null,
  evaluationTab: "metrics",
  saving: false,
  listPages: {},
  jobHistory: { total: 0, active_count: 0, latest_resume: null },
  predictionSource: "",
};
export async function api(path, options = {}) {
  const response = await fetch("/api" + path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      "X-Studio-Token": state.token || "",
      "X-Workspace": state.root || "",
      ...options.headers,
    },
  });
  if (!response.ok) {
    let message = await response.text();
    try {
      const data = JSON.parse(message);
      message =
        typeof data.detail === "string"
          ? data.detail
          : data.detail
              ?.map((x) => `${x.loc.slice(1).join(".")}: ${x.msg}`)
              .join("\n") || message;
    } catch {}
    throw new Error(message);
  }
  if (response.status === 204) return null;
  return response.json();
}
export function applyProject(data, reset = false) {
  const changed =
    state.root !== data.root ||
    JSON.stringify(state.project?.figures.map((f) => [f.id, f.name])) !==
      JSON.stringify(data.project.figures.map((f) => [f.id, f.name]));
  if (changed || reset) {
    state.scope = "all";
    state.stage = "video";
    state.selectedFramesFigure = null;
    state.datasetId = null;
    state.modelId = null;
    state.predictionSource = "";
    state.listPages = {};
  }
  data.project.scenes.asset_versions ??= {};
  Object.assign(state, {
    project: data.project,
    revision: data.revision,
    root: data.root,
    token: data.token,
  });
}
let saveQueue = Promise.resolve();
export async function save() {
  const snapshot = structuredClone(state.project);
  const work = saveQueue
    .catch(() => {})
    .then(async () => {
      state.saving = true;
      try {
        const data = await api("/project", {
          method: "PUT",
          body: JSON.stringify({ project: snapshot, revision: state.revision }),
        });
        state.revision = data.revision;
      } finally {
        state.saving = false;
      }
    });
  saveQueue = work;
  return work;
}
export function notice(message, error = false) {
  const el = document.createElement("div");
  el.className = "notice" + (error ? " error" : "");
  el.textContent = message;
  document.querySelector("#notifications").append(el);
  setTimeout(() => el.remove(), error ? 16000 : 5500);
}
export async function refresh() {
  const [artifacts, , videos] = await Promise.all([
    api("/artifacts"),
    loadJobs(),
    api("/videos"),
  ]);
  Object.assign(state, { artifacts, videos });
}
export async function loadJobs() {
  const data = await api(
    `/jobs?offset=${(state.listPages.jobs || 0) * 5}&limit=5`,
  );
  state.jobs = data.items;
  state.jobHistory = data;
  state.listPages.jobs = data.offset / data.limit;
  return data;
}
export function artifactUrl(a, path) {
  return `/api/artifacts/${a.id}/${path.split("/").map(encodeURIComponent).join("/")}`;
}
export const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (ch) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        ch
      ],
  );
export function merged(a, b) {
  const r = structuredClone(a);
  for (const [k, v] of Object.entries(b || {}))
    r[k] =
      v && typeof v === "object" && !Array.isArray(v)
        ? merged(r[k] || {}, v)
        : v;
  return r;
}
export function recipe() {
  return state.scope === "all"
    ? state.project.prepare.defaults
    : merged(
        state.project.prepare.defaults,
        state.project.prepare.overrides[state.scope],
      );
}
export function setPath(object, path, value) {
  const keys = path.split(".");
  let node = object;
  for (const key of keys.slice(0, -1)) node = node[key] ??= {};
  node[keys.at(-1)] = value;
}
export function getPath(object, path) {
  return path.split(".").reduce((n, k) => n?.[k], object);
}
export function scopedFigures() {
  return state.scope === "all"
    ? state.project.figures
    : state.project.figures.filter((f) => f.id === state.scope);
}
export const kindNames = {
  frames: "Кадры и маски",
  prepare: "Подготовка фигур",
  "prepare-candidates": "Три варианта 3D",
  pilot: "Пробный рендер",
  dataset: "Сборка датасета",
  train: "Обучение YOLO",
  evaluate: "Оценка качества",
  predict: "Поиск фигур",
  localize: "Координаты и 3D-сцена",
  install: "Настройка окружения",
  reuse: "Переиспользование модели",
};
export const statusNames = {
  queued: "В очереди",
  running: "Выполняется",
  success: "Завершено",
  failed: "Ошибка",
  cancelled: "Отменено",
  interrupted: "Прервано",
};
