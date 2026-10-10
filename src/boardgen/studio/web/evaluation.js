import { state, esc } from "./api.js";
import { renderLocalization, wireLocalization } from "./localization.js";
import {
  field,
  listPage,
  trainingContext,
  number,
  select,
  advanced,
  empty,
  choose,
  versionSelector,
  pathRow,
} from "./controls.js";
export function render() {
  const e = state.project.evaluation;
  const source = state.predictionSource;
  const model = state.artifacts.find((a) => a.id === state.modelId);
  const page = listPage(
    state.artifacts.filter((a) =>
      ["evaluate", "predict", "localize"].includes(a.kind),
    ),
    "evaluated",
    5,
  );
  return `<div class="card"><h2>Модель</h2>${versionSelector("train", state.modelId, "evaluation-model", "Результат обучения")}${model ? trainingContext(model.manifest.metadata) : ""}<div class="tabs" role="tablist">${[
    ["metrics", "Качество на разметке"],
    ["predict", "Свои фотографии и видео"],
    ["localize", "Координаты и 3D-сцена"],
  ]
    .map(
      ([k, t]) =>
        `<button role="tab" aria-selected="${state.evaluationTab === k}" class="${state.evaluationTab === k ? "active" : ""}" data-evaluation-tab="${k}">${t}</button>`,
    )
    .join("")}</div>${
    state.evaluationTab === "metrics"
      ? `${versionSelector("dataset", state.datasetId, "evaluation-dataset", "Датасет с разметкой")}<div class="grid">${field(
          "Раздел для оценки",
          select("evaluation.split", e.split, [
            ["val", "Val — настройка модели"],
            ["test", "Test — независимая проверка"],
          ]),
        )}${field("Уверенность при разборе ошибок", number("evaluation.confidence", e.confidence, 0.001, 0.999, 0.01), "confidence")}</div>${advanced(`<div class="grid">${field("Размер входа, px", `<input data-bind="evaluation.imgsz" type="number" min="64" max="2048" step="32" value="${e.imgsz ?? ""}" placeholder="Как при обучении">`, "inferenceSize")}${field("IoU совпадения с разметкой", number("evaluation.matching_iou", e.matching_iou, 0.01, 0.99, 0.01), "matching")}${field("IoU удаления дублей (NMS)", number("evaluation.nms_iou", e.nms_iou, 0.01, 0.99, 0.01), "nms")}</div>`)}<button class="primary" data-action="evaluate" ${!state.modelId || !state.datasetId ? "disabled" : ""}>Оценить качество</button>`
      : state.evaluationTab === "localize"
        ? renderLocalization()
        : `${field("Фото, видео или папка изображений", pathRow(source, "prediction-source"))}<div class="actions"><button class="small" data-action="prediction-folder">Выбрать папку</button></div><div class="grid" style="margin-top:20px">${field("Уверенность", number("evaluation.confidence", e.confidence, 0.001, 0.999, 0.01), "confidence")}${field("IoU удаления дублей (NMS)", number("evaluation.nms_iou", e.nms_iou, 0.01, 0.99, 0.01), "nms")}</div><p class="hint">На своих файлах показываются рамки и уверенность. Метрики AP требуют правильную разметку.</p><button class="primary" data-action="predict" ${!state.modelId || !source ? "disabled" : ""}>Найти фигуры</button>`
  }</div><div class="card"><h2>Проверки и предсказания</h2>${
    page.rows
      .map(
        (a) =>
          `<div class="job-head" style="margin-bottom:14px"><div><strong>${a.kind === "evaluate" ? "Оценка: " + a.manifest.metadata.split : a.kind === "localize" ? "Координаты и 3D-сцена" : "Поиск на своих файлах"}</strong><p class="hint">${new Date(a.created).toLocaleString("ru")} · ${a.id.slice(0, 6)}</p></div><button class="small" data-artifact="${a.id}">Посмотреть</button></div>`,
      )
      .join("") ||
    empty("Запустите оценку на разметке или поиск на своих файлах.")
  }${page.navigation}</div>`;
}
export function wire(redraw, run, showArtifact, onChange) {
  wireLocalization(redraw, run, onChange);
  document.querySelector("#evaluation-model").onchange = (e) => {
    state.modelId = e.target.value;
    redraw();
  };
  document
    .querySelector("#evaluation-dataset")
    ?.addEventListener("change", (e) => {
      state.datasetId = e.target.value;
      redraw();
    });
  document.querySelectorAll("[data-evaluation-tab]").forEach(
    (b) =>
      (b.onclick = () => {
        state.evaluationTab = b.dataset.evaluationTab;
        redraw();
      }),
  );
  document
    .querySelector("[data-action=evaluate]")
    ?.addEventListener("click", () =>
      run("evaluate", { dataset_id: state.datasetId, model_id: state.modelId }),
    );
  document
    .querySelector("[data-action=predict]")
    ?.addEventListener("click", () =>
      run("predict", {
        model_id: state.modelId,
        source: state.predictionSource,
      }),
    );
  for (const action of ["prediction-source", "prediction-folder"])
    document
      .querySelector(`[data-action=${action}]`)
      ?.addEventListener("click", async () => {
        const path = await choose({
          title: "Файлы для поиска фигур",
          mode: action === "prediction-folder" ? "folder" : "file",
          extensions: [
            ".jpg",
            ".jpeg",
            ".png",
            ".mp4",
            ".mov",
            ".mkv",
            ".avi",
            ".webp",
          ],
        });
        if (path) {
          state.predictionSource = path;
          if (state.evaluationTab === "localize") {
            state.project.localization.corners = [];
            onChange();
          }
          redraw();
        }
      });
  document
    .querySelectorAll("[data-artifact]")
    .forEach((b) => (b.onclick = () => showArtifact(b.dataset.artifact)));
}
