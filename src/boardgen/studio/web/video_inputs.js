import { state, api, applyProject, refresh, notice, esc } from "./api.js";
import { choose, showResult } from "./controls.js";

async function chooseMode(path) {
  if (!state.project.figures.length) return "replace";
  return new Promise((resolve) => {
    const dialog = showResult(
      "Подключить видео",
      `<p class="path-value">${esc(path)}</p><p>В проекте уже ${state.project.figures.length} видео. Как подключить новую папку?</p><p class="hint">При замене видео вне новой папки и их индивидуальные настройки будут убраны из списка. Общие параметры сохранятся. Исходные файлы и история результатов остаются на диске.</p><div class="actions"><button class="primary" data-video-mode="replace">Заменить список</button><button data-video-mode="add">Добавить к текущим</button><button data-video-mode="cancel">Отмена</button></div>`,
    );
    dialog.addEventListener("close", () => resolve(null), { once: true });
    dialog.querySelectorAll("[data-video-mode]").forEach(
      (button) =>
        (button.onclick = () => {
          resolve(
            button.dataset.videoMode === "cancel"
              ? null
              : button.dataset.videoMode,
          );
          dialog.close();
        }),
    );
  });
}
export async function connectVideoFolder(flush, redraw) {
  const path = await choose({ title: "Папка с видео" });
  if (!path) return;
  const mode = await chooseMode(path);
  if (!mode) return;
  await flush();
  applyProject(
    await api("/videos", {
      method: "POST",
      body: JSON.stringify({ path, mode }),
    }),
  );
  state.listPages.videos = 0;
  state.listPages.prepared = 0;
  await refresh();
  redraw();
  notice(mode === "replace" ? "Список видео заменён" : "Новые видео добавлены");
}
