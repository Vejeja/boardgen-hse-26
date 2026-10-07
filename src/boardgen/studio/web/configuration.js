import { state, api, applyProject, esc, merged, notice } from "./api.js";
import { showResult } from "./controls.js";

export async function editConfiguration(redraw, flush, artifact = null) {
  await flush();
  const figure =
    artifact && state.project.figures.find((f) => f.id === artifact.figure_id);
  const current = figure
    ? merged(
        state.project.prepare.defaults,
        state.project.prepare.overrides[figure.id],
      )
    : state.project;
  const saved = artifact?.manifest.metadata.recipe;
  const title = figure
    ? `Рецепт новой подготовки — ${figure.name}`
    : "Редактирование конфигурации проекта";
  const dialog = showResult(
    title,
    `${artifact ? `<details><summary>Параметры результата ${esc(artifact.id.slice(0, 8))} (только чтение)</summary>${saved ? `<pre>${esc(JSON.stringify(saved, null, 2))}</pre>` : '<p class="hint">Исходный рецепт импортированной модели не сохранён. Ниже — текущие настройки проекта для нового запуска.</p>'}</details>` : ""}<p class="hint">JSON проверяется перед сохранением. Изменение настроек не запускает вычисления и не меняет готовые модели, датасеты или выбор.</p><label class="field"><span class="label">${figure ? "Рецепт подготовки" : "Конфигурация"}</span><textarea id="config-json" rows="22" spellcheck="false">${esc(JSON.stringify(current, null, 2))}</textarea></label><p id="config-error" class="error-text" role="alert"></p><div class="actions"><button class="primary" id="save-config">Сохранить настройки</button><button id="cancel-config">Отмена</button></div>`,
  );
  dialog.querySelector("#cancel-config").onclick = () => dialog.close();
  dialog.querySelector("#save-config").onclick = async (event) => {
    event.target.disabled = true;
    try {
      const value = JSON.parse(dialog.querySelector("#config-json").value);
      const project = figure ? structuredClone(state.project) : value;
      if (figure) {
        project.prepare.overrides[figure.id] = value;
        const previous = current;
        if (
          JSON.stringify(previous.matte) !== JSON.stringify(value.matte) ||
          previous.shape.candidate_frames !== value.shape?.candidate_frames ||
          previous.shape.frame_size !== value.shape?.frame_size
        ) {
          const f = project.figures.find((f) => f.id === figure.id);
          f.reviewed_views = {};
          f.reviewed_frames_id = null;
        }
      }
      const data = await api("/project", {
        method: "PUT",
        body: JSON.stringify({ project, revision: state.revision }),
      });
      applyProject(data);
      dialog.close();
      redraw();
      notice("Настройки сохранены. Готовые результаты сохранены.");
    } catch (error) {
      dialog.querySelector("#config-error").textContent =
        error instanceof SyntaxError
          ? `Ошибка JSON: ${error.message}`
          : error.message;
    } finally {
      event.target.disabled = false;
    }
  };
}
