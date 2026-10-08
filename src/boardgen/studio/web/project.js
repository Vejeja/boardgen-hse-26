import { state, esc, api, applyProject, refresh, notice } from "./api.js";
import { connectVideoFolder } from "./video_inputs.js";
import {
  field,
  input,
  pathRow,
  help,
  choose,
  showResult,
  empty,
} from "./controls.js";
export function render() {
  const p = state.project;
  return `<div class="card"><h2>Папка и настройки</h2><div class="grid">${field("Название проекта", input("name", p.name))}${field("Папка проекта", pathRow(state.root, "workspace"), "", "Здесь сохраняются project.yaml, задания и результаты.")}</div><div class="actions"><button data-action="import-config">Загрузить project.yaml</button><button data-action="export-config">Скачать настройки</button>${help("config")}<input hidden id="config-file" type="file" accept=".yaml,.yml"></div></div><div class="card"><div class="actions spaced"><h2 style="margin:0">Исходные видео ${help("video")}</h2><button data-action="connect-videos">Выбрать папку</button></div><p class="muted">Одно видео — одна фигура с разных сторон. Общие настройки действуют на всю папку.</p>${p.figures.length ? `<p class="badge">Подключено видео: ${p.figures.length}</p><p class="small-text muted">Названия фигур и фрагменты видео настраиваются в следующем разделе.</p>` : empty("Подключите папку с видео фигур.")}</div><div class="card"><div class="actions spaced"><h2 style="margin:0">Окружение ${help("environment")}</h2><button data-action="environment">Проверить окружение</button></div><div id="environment-result"></div></div><div class="actions"><button class="primary" data-action="next-prepare">Далее: подготовка фигур →</button></div>`;
}
export function wire(navigate, redraw, run, flush) {
  document.querySelector("[data-action=workspace]").onclick = async () => {
    const path = await choose({ title: "Папка проекта" });
    if (!path) return;
    await flush();
    applyProject(
      await api("/workspace", {
        method: "POST",
        body: JSON.stringify({ path }),
      }),
    );
    await refresh();
    state.scope = "all";
    state.datasetId = null;
    state.modelId = null;
    redraw();
  };
  document.querySelector("[data-action=export-config]").onclick = async () => {
    await flush();
    const link = document.createElement("a");
    link.href = "/api/config";
    link.download = "project.yaml";
    link.click();
  };
  document.querySelector("[data-action=connect-videos]").onclick = () =>
    connectVideoFolder(flush, redraw);
  document.querySelector("[data-action=import-config]").onclick = () =>
    document.querySelector("#config-file").click();
  document.querySelector("#config-file").onchange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      await flush();
      applyProject(
        await api("/config", {
          method: "POST",
          body: JSON.stringify({ content: await file.text() }),
        }),
        true,
      );
      await refresh();
      state.scope = "all";
      redraw();
      notice("Настройки загружены");
    } catch (e) {
      notice(e.message, true);
    }
  };
  document.querySelector("[data-action=environment]").onclick = async () => {
    await flush();
    const result = await api("/environment");
    document.querySelector("#environment-result").innerHTML =
      `<div class="callout ${result.ready ? "" : "warning"}">${
        result.ready
          ? "Окружение готово."
          : "Требуется настройка: " +
            result.items
              .filter((i) => !i.ready)
              .map((i) => esc(i.name))
              .join(", ")
      } </div><details><summary>Диагностика</summary><p class="small-text muted">Python ${esc(result.python)} · ${esc(result.executable)}</p><table><thead><tr><th>Компонент</th><th>Состояние</th><th>Версия или путь</th></tr></thead><tbody>${result.items.map((i) => `<tr><td>${esc(i.name)}</td><td>${i.ready ? "Готов" : "Отсутствует"}</td><td>${esc(i.version || i.path)}</td></tr>`).join("")}</tbody></table></details>`;
  };
  document.querySelector("[data-action=next-prepare]").onclick = () =>
    navigate("prepare");
}
