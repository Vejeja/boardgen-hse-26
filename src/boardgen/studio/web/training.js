import { state, esc } from "./api.js";
import {
  listPage,
  trainingContext,
  field,
  number,
  select,
  check,
  advanced,
  empty,
  choose,
  versionSelector,
} from "./controls.js";
export function render() {
  const t = state.project.training;
  const selected = state.artifacts.find((a) => a.id === state.datasetId);
  const page = listPage(
    state.artifacts.filter((a) => a.kind === "train"),
    "trained",
    5,
  );
  const resume = state.jobHistory.latest_resume;
  return `<div class="card"><h2>Данные и модель</h2>${versionSelector("dataset", state.datasetId, "training-dataset", "Версия датасета")}${
    selected
      ? `<div class="callout">${Object.entries(
          selected.manifest.metadata.counts,
        )
          .map(([s, n]) => `${s}: ${n}`)
          .join(
            " · ",
          )} · классов: ${selected.manifest.metadata.names.length}</div>`
      : ""
  }<div class="grid">${field("Начальные веса", select("training.weights", t.weights, [...["n", "s", "m", "l", "x"].map((v) => [`yolo11${v}.pt`, `YOLO11${v}`]), ...(!/^yolo11[nsm lx]\.pt$/.test(t.weights) ? [[t.weights, t.weights.split("/").at(-1)]] : [])]), "model")}<div class="field"><span class="label">Локальные веса detect</span><button data-action="training-weights">Выбрать .pt</button></div>${field("Эпохи", number("training.epochs", t.epochs, 1, 10000))}${field("Размер входа, px", number("training.imgsz", t.imgsz, 64, 2048, 32))}</div>${selected?.manifest.metadata.near_duplicate_count ? `<div class="callout warning">Найдено похожих пар между разделами: ${selected.manifest.metadata.near_duplicate_count}. Просмотрите их в датасете перед обучением.</div><label class="check-row"><input type="checkbox" id="accept-near">Похожие пары просмотрены; разделение принимаю</label>` : ""}${advanced(
    `<div class="grid three">${field("Batch", number("training.batch", t.batch, 1, 128), "batch")}${field("Остановка без улучшения, эпох", number("training.patience", t.patience, 0, 1000))}${field("Процессов загрузки данных", number("training.workers", t.workers, 0, 16))}${field(
      "Оптимизатор",
      select("training.optimizer", t.optimizer, [
        ["auto", "Auto"],
        ["SGD", "SGD"],
        ["Adam", "Adam"],
        ["AdamW", "AdamW"],
      ]),
    )}${field("Начальная скорость обучения", number("training.lr0", t.lr0, 0.00001, 1, 0.0001), "learningRate")}${field("Seed", number("training.seed", t.seed, 0, 2147483647), "seed")}${field("Вероятность Mosaic", number("training.mosaic", t.mosaic, 0, 1, 0.05))}</div>${check("training.deterministic", t.deterministic, "Детерминированные операции, где поддерживаются")}${check("training.cache", t.cache, "Кэшировать изображения в памяти")}`,
  )}<p class="hint">Устройство выбирается автоматически: CUDA, Apple MPS или CPU. Оно будет записано в результатах запуска.</p><button class="primary" data-action="train" ${!state.datasetId ? "disabled" : ""}>Начать обучение</button></div><div class="card"><h2>Продолжить обучение</h2>${resume ? `<p>Последний остановленный запуск: <strong>${resume.id.slice(0, 6)}</strong> · ${new Date(resume.created).toLocaleString("ru")}</p><p class="hint">Датасет ${resume.parameters.dataset_id.slice(0, 6)} · ${resume.training.epochs} эпох · вход ${resume.training.imgsz} px. Продолжение использует сохранённые настройки и состояние оптимизатора.</p><button data-resume="${resume.id}">Продолжить с сохранённой эпохи</button><details><summary>Что хранится на диске?</summary><p class="hint">Настройки: ${(resume.settings_bytes / 1024).toFixed(1)} КБ; checkpoint: ${(resume.checkpoint_bytes / 1024 / 1024).toFixed(1)} МБ. Веса модели загружаются в память при продолжении. Другие незавершённые запуски доступны в истории заданий.</p></details>` : empty("Нет незавершённого обучения с сохранённой эпохой.")}</div><div class="card"><h2>Результаты обучения</h2>${
    page.rows
      .map(
        (a) =>
          `<div class="job-head" style="margin-bottom:15px"><div><strong>YOLO · ${a.id.slice(0, 6)}</strong><p class="hint">${esc(a.manifest.metadata.device)} · ${a.manifest.metadata.epoch_count} эпох · ${new Date(a.created).toLocaleString("ru")}</p>${trainingContext(a.manifest.metadata)}</div><button data-artifact="${a.id}" class="small">Метрики и веса</button></div>`,
      )
      .join("") || empty("Графики, best.pt и last.pt появятся после обучения.")
  }${page.navigation}</div>`;
}
export function wire(redraw, run, showArtifact, onChange) {
  document.querySelector("#training-dataset").onchange = (e) => {
    state.datasetId = e.target.value;
    redraw();
  };
  document.querySelector("[data-action=training-weights]").onclick =
    async () => {
      const path = await choose({
        title: "Локальные веса YOLO detect",
        mode: "file",
        extensions: [".pt"],
      });
      if (path) {
        state.project.training.weights = path;
        onChange(true);
      }
    };
  document.querySelector("[data-action=train]").onclick = () =>
    run("train", {
      dataset_id: state.datasetId,
      accept_near_duplicates: !!document.querySelector("#accept-near")?.checked,
    });
  document
    .querySelectorAll("[data-resume]")
    .forEach(
      (b) =>
        (b.onclick = () => run("train", { resume_job_id: b.dataset.resume })),
    );
  document
    .querySelectorAll("[data-artifact]")
    .forEach((b) => (b.onclick = () => showArtifact(b.dataset.artifact)));
}
