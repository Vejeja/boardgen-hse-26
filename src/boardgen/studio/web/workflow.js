import { state, api, esc, scopedFigures } from "./api.js";
import { chosenFigures, datasetMatches } from "./workflow_data.js";

export function renderWorkflow() {
  const chosen = chosenFigures(state.project);
  const pinned = chosen.filter((f) =>
    state.artifacts.some(
      (a) =>
        a.kind === "figure" &&
        a.id === state.project.scenes.asset_versions[f.id] &&
        a.figure_id === f.id,
    ),
  );
  const datasets = state.artifacts.filter((a) => a.kind === "dataset"),
    models = state.artifacts.filter((a) => a.kind === "train");
  const latest = datasets[0];
  const same =
    latest &&
    datasetMatches(state.project, latest.manifest.metadata.workflow_inputs);
  return `<section class="card workflow" aria-label="Готовность проекта"><div class="actions spaced"><strong>Видео → подготовка → выбор → сцены → обучение → проверка</strong><button class="small" data-config-project>Полная конфигурация</button></div><p class="hint">Можно просматривать настройки и результаты в любом порядке. Вычисления запускаются отдельными кнопками и используют сохранённые входы; переход между страницами ничего не вычисляет.</p><div class="workflow-facts"><span>Видео: ${state.project.figures.length}</span><span>Закреплено: ${pinned.length} / ${chosen.length}</span><span>Датасеты: ${datasets.length}</span><span>Модели: ${models.length}</span></div><p class="hint">${!pinned.length ? "Следующий шаг: подготовьте кандидатов и закрепите версии в сравнении." : pinned.length !== chosen.length ? "Часть выбранных классов не имеет доступной версии: откройте сравнение." : !latest ? "Модели выбраны. Проверьте пробный рендер и соберите датасет." : same ? "Последний датасет соответствует текущим входам генерации. Для обучения выберите его сохранённую версию." : "Настройки или выбор отличаются от последнего датасета. Он доступен для воспроизведения; новый набор требует отдельной сборки."}</p><p id="readiness-reason" class="hint" role="status"></p></section>`;
}
let revision = 0;
export async function updateReadiness() {
  const current = ++revision;
  const requests = [];
  for (const kind of [
    "frames",
    "prepare",
    "prepare-candidates",
    "pilot",
    "dataset",
    "train",
    "evaluate",
    "predict",
    "localize",
  ]) {
    const button = document.querySelector(`[data-action="${kind}"]`);
    if (!button) continue;
    const parameters = ["frames", "prepare", "prepare-candidates"].includes(
      kind,
    )
      ? { figure_ids: scopedFigures().map((f) => f.id) }
      : kind === "train"
        ? {
            dataset_id: state.datasetId,
            accept_near_duplicates:
              !!document.querySelector("#accept-near")?.checked,
          }
        : kind === "evaluate"
          ? { dataset_id: state.datasetId, model_id: state.modelId }
          : ["predict", "localize"].includes(kind)
            ? { model_id: state.modelId, source: state.predictionSource }
            : {};
    button.disabled = true;
    requests.push(
      (async () => {
        let result;
        try {
          result = await api("/preflight", {
            method: "POST",
            body: JSON.stringify({ kind, parameters }),
          });
        } catch (error) {
          result = { ready: false, reason: error.message };
        }
        if (current !== revision || !button.isConnected) return null;
        button.disabled = !result.ready;
        button.title =
          result.reason || "Входы проверены; запуск создаст отдельное задание";
        return result.ready ? null : `${button.textContent}: ${result.reason}`;
      })(),
    );
  }
  const reasons = (await Promise.all(requests)).filter(Boolean);
  const root = document.querySelector("#readiness-reason");
  if (current === revision && root) root.textContent = reasons.join(" · ");
}
