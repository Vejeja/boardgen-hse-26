import { api, state, applyProject, esc, notice, refresh } from "./api.js";
import { comparisonMarkup } from "./comparison_markup.js";
import { mountComparison } from "./comparison.js";
let stop = null;
let generation = 0;
export function disposeComparison() {
  ++generation;
  stop?.();
  stop = null;
}
export function renderComparison() {
  return `<h2>Сравнение и выбор</h2><p class="hint">Это единственное место выбора моделей для новых сцен. Выбор сохраняется в проекте: одна точная версия на класс. Новые вычисления его не заменяют; уже собранные датасеты сохраняют прежние входы.</p><details><summary>Подключить сохранённые кандидаты</summary><div class="path-row"><select id="comparison-package"><option value="">Выберите набор</option></select><button id="connect-candidates">Подключить</button></div><p class="hint">Подключаются варианты для совпадающих исходных видео. Текущий выбор сохранится. Прежний JSON выбора можно импортировать на странице проекта.</p></details><div id="figure-comparison"><div class="toolbar"><label>Фигура <select id="object"></select></label></div>${comparisonMarkup}</div>`;
}
export async function wireComparison(redraw, flush, showArtifact) {
  const selector = document.querySelector("#comparison-package");
  const root = document.querySelector("#figure-comparison");
  const current = ++generation;
  try {
    const packages = await api("/comparison/packages");
    if (current !== generation || !root.isConnected) return;
    for (const pkg of packages) selector.add(new Option(pkg.name, pkg.path));
    document.querySelector("#connect-candidates").onclick = async (event) => {
      if (!selector.value) return notice("Выберите сохранённый набор", true);
      event.target.disabled = true;
      try {
        await flush();
        const data = await api("/candidate-library", {
          method: "POST",
          body: JSON.stringify({ path: selector.value }),
        });
        await refresh();
        redraw();
        notice(`Доступно кандидатов: ${data.count}. Выбор сохранён.`);
      } catch (error) {
        notice(error.message, true);
      } finally {
        if (event.target.isConnected) event.target.disabled = false;
      }
    };
    const catalog = await api("/comparison/catalog");
    if (current !== generation || !root.isConnected) return;
    if (!catalog.length) {
      root.textContent = "Подключите видео на странице проекта.";
      return;
    }
    let choices = [];
    for (const [object, id] of Object.entries(
      state.project.scenes.asset_versions,
    )) {
      if (
        state.project.scenes.selected_classes !== null &&
        !state.project.scenes.selected_classes.includes(object)
      )
        continue;
      const a = await api(`/artifacts/${id}`);
      choices.push({
        object,
        id,
        name: state.project.figures.find((f) => f.id === object)?.name,
        mesh_sha256: a.manifest.files[a.manifest.metadata.mesh || "figure.glb"],
        source_video_sha256: a.manifest.metadata.source_sha256,
        label: a.manifest.metadata.research_title || "Выбранная версия",
      });
    }
    if (current !== generation || !root.isConnected) return;
    stop = mountComparison(
      root,
      catalog,
      "project",
      choices,
      async (next) => {
        await flush();
        const changed = next.find(
          (c) => choices.find((old) => old.object === c.object)?.id !== c.id,
        );
        const removed = choices.find(
          (c) => !next.some((n) => n.object === c.object),
        );
        const object = changed?.object || removed?.object;
        if (!object) return;
        const data = await api("/candidate-choice", {
          method: "PUT",
          body: JSON.stringify({
            figure_id: object,
            artifact_id: changed?.id || null,
            revision: state.revision,
          }),
        });
        applyProject(data);
        choices = next;
        window.dispatchEvent(new Event("workflow-change"));
        notice("Выбор сохранён в проекте");
      },
      showArtifact,
    );
  } catch (error) {
    if (root.isConnected)
      root.innerHTML = `<p class="error-text">${esc(error.message)}</p>`;
  }
}
