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
  const importChoice = document.createElement("div");
  importChoice.className = "card";
  importChoice.innerHTML = `<h2>Импорт прежнего выбора моделей</h2><p class="muted">Сравнение находится в «Подготовка фигур → Сравнение и выбор». Здесь можно импортировать прежний JSON «Мой выбор». Будет создан отдельный проект только с отмеченными фигурками и моделями.</p><label>Набор моделей <select id="selection-package"></select></label><div class="actions"><button id="import-selection" disabled>Импортировать выбор для YOLO</button><input hidden id="selection-file" type="file" accept=".json,application/json"></div><p class="hint">После импорта проверьте размеры фигур и запустите пробный рендер в разделе «Датасет». Начальная высота 5 см условная; её можно изменить. Обучение запускается отдельно.</p>`;
  document
    .querySelector("[data-action=next-prepare]")
    .parentElement.before(importChoice);
  let pendingSelection = null;
  try {
    const value = new URLSearchParams(location.hash.slice(1)).get("selection");
    if (value) {
      pendingSelection = JSON.parse(value);
      const preview = document.createElement("p");
      preview.textContent =
        "Передано из галереи: " +
        pendingSelection.choices.map((choice) => choice.name).join("; ");
      const button = document.createElement("button");
      button.id = "import-pending-selection";
      button.textContent = `Импортировать выбранные фигурки (${pendingSelection.choices.length})`;
      button.disabled = true;
      button.onclick = () => importChosen(pendingSelection, button);
      importChoice.append(preview, button);
    }
  } catch (error) {
    notice("Не удалось прочитать выбор: " + error.message, true);
  }
  api("/figure-selection/packages")
    .then((packages) => {
      if (!importChoice.isConnected) return;
      const select = importChoice.querySelector("#selection-package");
      select.replaceChildren(
        ...packages.map((path) => new Option(path.split("/").at(-1), path)),
      );
      importChoice.querySelector("#import-selection").disabled =
        !packages.length;
      const pendingButton = importChoice.querySelector(
        "#import-pending-selection",
      );
      if (pendingButton) pendingButton.disabled = !packages.length;
      if (!packages.length)
        select.replaceChildren(new Option("Нет подготовленных наборов", ""));
    })
    .catch((error) => notice(error.message, true));
  importChoice.querySelector("#import-selection").onclick = () =>
    importChoice.querySelector("#selection-file").click();
  importChoice.querySelector("#selection-file").onchange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      await importChosen(
        JSON.parse(await file.text()),
        importChoice.querySelector("#import-selection"),
      );
    } catch (error) {
      notice(error.message, true);
    }
    event.target.value = "";
  };
  async function importChosen(selection, button) {
    button.disabled = true;
    try {
      await flush();
      const data = await api("/figure-selection", {
        method: "POST",
        body: JSON.stringify({
          selection,
          package: importChoice.querySelector("#selection-package").value,
        }),
      });
      applyProject(data, true);
      history.replaceState(null, "", location.pathname + location.search);
      await refresh();
      navigate("dataset");
      notice(
        `Импортировано фигур: ${data.selection_import.class_count}. Новый проект готов к пробному рендеру.`,
      );
    } catch (error) {
      notice(error.message, true);
      button.disabled = false;
    }
  }
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
      } ${!result.ready ? '<button class="small" id="install-environment">Установить недостающее</button>' : ""}</div><details><summary>Диагностика</summary><p class="small-text muted">Python ${esc(result.python)} · ${esc(result.executable)}</p><table><thead><tr><th>Компонент</th><th>Состояние</th><th>Версия или путь</th></tr></thead><tbody>${result.items.map((i) => `<tr><td>${esc(i.name)}</td><td>${i.ready ? "Готов" : "Отсутствует"}</td><td>${esc(i.version || i.path)}</td></tr>`).join("")}</tbody></table></details>`;
    document
      .querySelector("#install-environment")
      ?.addEventListener("click", () => run("install"));
  };
  document.querySelector("[data-action=next-prepare]").onclick = () =>
    navigate("prepare");
}
