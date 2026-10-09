import { api, state, esc, notice } from "./api.js";
import { pageWindow } from "./lists.js";
const tips = {
  video:
    "Снимите одну фигуру с разных сторон. Настройки папки применяются ко всем видео; для отдельных фигур можно задать исключения.",
  config:
    "Скачиваются текущие настройки, включая значения по умолчанию: видео, фигуры, обработка, сцены и обучение. Вычисленные файлы хранятся отдельно.",
  environment:
    "Проверка покажет недостающие программы. Кнопка «Установить недостающее» запустит их настройку.",
  scope:
    "Общие настройки действуют для всех фигур. Выберите одну фигуру, чтобы изменить только её параметры.",
  clip: "По умолчанию используется всё видео. Здесь можно убрать начало или конец только этого файла.",
  matte:
    "BiRefNet выделяет фигуру и сохраняет прозрачный фон. Классический метод сравнивает кадры со статичным фоном.",
  maskSize:
    "Размер изображения на входе BiRefNet. Большое разрешение лучше сохраняет тонкие детали, но требует больше памяти.",
  crossCheck:
    "Сравнивает BiRefNet с классической маской и показывает расхождения. Основная маска остаётся прежней.",
  frames:
    "Кадры равномерно выбираются из выбранного фрагмента. Каждый пятый остаётся для проверки и не участвует в построении модели.",
  views:
    "Выберите переднюю, левую, тыльную и правую стороны фигуры в одном обороте. Порядок кадров сам по себе не определяет сторону.",
  mesh: "Разрешение извлечения поверхности MV. Более высокое значение сохраняет мелкие детали и увеличивает расход памяти.",
  cameras:
    "Orbit использует неподвижную камеру и вращение фигуры. SfM восстанавливает положение камеры по совпадающим деталям (SIFT).",
  period:
    "Количество выбранных кадров на один полный оборот. Передний ракурс и период задают начальное направление; затем камеры уточняются.",
  focal:
    "Начальная фокусная длина в пикселях подготовленного кадра. Её можно взять из калибровки камеры; приблизительное значение требует проверки наложений.",
  texture:
    "Фототекстура переносит цвета и детали из видео на 3D-поверхность. Текстура реконструкции использует цвет, восстановленный Spirula.",
  registration:
    "Силуэты: каждый ракурс подгоняется к форме. Orbit: одна общая регистрация к восстановленным камерам, без отдельных поправок поворота.",
  photoViews:
    "Дополнительные фото выбираются по последовательным четырём сторонам в одном обороте. Если кадры взяты из разных оборотов, используйте 4 выбранных фото.",
  atlas:
    "Максимальная сторона карты текстуры в пикселях. Большая карта даёт больше цветовых деталей, если исходные кадры достаточно чёткие.",
  board:
    "Изображение поверхности стола или игрового поля. Ширина и длина задают его физический размер в сцене.",
  skybox:
    "Панорама окружающего помещения задаёт фон и освещение. Можно выбрать несколько файлов: генератор будет менять их между сценами.",
  camera:
    "Камера смотрит на центр поля. Расстояние измеряется до центра; угол над полем — от горизонтали, азимут — направление вокруг поля.",
  seed: "Фиксирует случайный выбор сцен и разделение данных. Повторяемость численных вычислений зависит также от версии backend и устройства.",
  relativeScale:
    "Оценка по маскам сохраняет соотношение видимых размеров. Она подходит только для одинаковых камеры, расстояния и масштаба съемки; абсолютную высоту задайте по измерению одной фигуры.",
  scale:
    "Базовая высота используется, когда физический размер неизвестен. Измеренную высоту конкретной фигуры можно задать отдельно.",
  split:
    "Train используется для обучения, val — для настройки. Кадры одной группы остаются в одном разделе. Test берётся только из отдельного размеченного источника.",
  groups:
    "CSV со столбцами image,group. Путь image относительно корня YOLO-датасета; одинаковый group объединяет кадры одной съёмки.",
  negative:
    "Отсутствующий файл разметки будет считаться пустым изображением только после этого подтверждения. Не включайте для неразмеченных фигур.",
  confidence:
    "Минимальная уверенность для показа предсказаний и разбора ошибок. AP считается отдельно по всем порогам уверенности.",
  matching:
    "Минимальное пересечение рамок (IoU), при котором предсказание считается совпадением с разметкой при разборе ошибок.",
  nms: "Порог IoU для удаления повторных предсказаний. Это отдельная настройка от сопоставления с разметкой.",
  batch:
    "Число изображений за один шаг обучения. Если не хватает памяти, уменьшите значение.",
  inferenceSize:
    "По умолчанию используется размер входа выбранной модели при обучении. Другое значение позволяет сравнить скорость и качество распознавания.",
  learningRate:
    "Начальный шаг оптимизатора. При Auto Ultralytics подбирает его автоматически; для ручной настройки выберите SGD, Adam или AdamW.",
  model:
    "YOLO11n — самая лёгкая модель из этой серии. s/m/l/x используют больше памяти; также можно выбрать локальные веса detect (.pt).",
};
export function help(key) {
  return `<button type="button" class="help" data-help="${esc(key)}" aria-label="Подсказка" aria-expanded="false">?</button>`;
}
let fieldIndex = 0;
export function field(label, control, key = "", hint = "") {
  if (control.includes('class="range"')) {
    const bounds = [...control.matchAll(/<span>(.*?)<\/span>/g)].map(
      (m) => m[1],
    );
    let boundIndex = 0;
    control = control.replace(/<input\b[^>]*>/g, (tag) =>
      tag.replace(
        /^<input/,
        `<input aria-label="${esc(label + " " + bounds[boundIndex++])}"`,
      ),
    );
  }
  const tag = control.match(/<(input|select|button)\b[^>]*>/);
  let id = tag?.[0].match(/\bid="([^"]+)"/)?.[1];
  if (tag && !id) {
    id = `field-${++fieldIndex}`;
    control = control.replace(tag[0], tag[0].replace(/^<\w+/, `$& id="${id}"`));
  }
  return `<div class="field"><span class="label">${id ? `<label for="${id}">${label}</label>` : label}${key ? help(key) : ""}</span>${control}${hint ? `<span class="hint">${hint}</span>` : ""}</div>`;
}
export function input(path, value, type = "text", extra = "") {
  return `<input data-bind="${path}" type="${type}" value="${esc(value)}" ${extra}>`;
}
export function number(path, value, min, max, step = 1) {
  return input(
    path,
    value,
    "number",
    `min="${min}" max="${max}" step="${step}"`,
  );
}
export function select(path, value, options) {
  return `<select data-bind="${path}">${options.map(([v, t]) => `<option value="${esc(v)}" ${String(v) === String(value) ? "selected" : ""}>${t}</option>`).join("")}</select>`;
}
export function check(path, value, label, key = "") {
  return `<label class="check-row"><input type="checkbox" data-bind="${path}" ${value ? "checked" : ""}>${label}${key ? ` ${help(key)}` : ""}</label>`;
}
export function range(
  path,
  values,
  min,
  max,
  step = 1,
  unit = "",
  labels = ["от", "до"],
) {
  return `<div class="range">${values.map((v, i) => `<label><span>${labels[i]}</span>${number(path + "." + i, v, min, max, step)}${unit}</label>`).join("")}</div>`;
}
export function empty(message) {
  return `<div class="empty">${message}</div>`;
}
export function pagination(key, total, size = 5) {
  const p = pageWindow(total, state.listPages[key] || 0, size);
  state.listPages[key] = p.page;
  if (total <= size) return "";
  return `<nav class="pagination" aria-label="Страницы списка"><span>${p.start + 1}–${p.end} из ${total}</span><div class="actions"><button class="small" data-list-key="${key}" data-list-page="${p.page - 1}" ${p.page === 0 ? "disabled" : ""}>Назад</button><span>Страница ${p.page + 1} из ${p.pages}</span><button class="small" data-list-key="${key}" data-list-page="${p.page + 1}" ${p.page + 1 === p.pages ? "disabled" : ""}>Далее</button></div></nav>`;
}
export function listPage(items, key, size = 5) {
  const p = pageWindow(items.length, state.listPages[key] || 0, size);
  state.listPages[key] = p.page;
  return {
    rows: items.slice(p.start, p.end),
    navigation: pagination(key, items.length, size),
  };
}
export function wirePages(root, redraw) {
  root.querySelectorAll("[data-list-key]").forEach((button) => {
    button.onclick = async () => {
      state.listPages[button.dataset.listKey] = Number(button.dataset.listPage);
      await redraw(button.dataset.listKey);
    };
  });
}
export function trainingContext(metadata) {
  const dataset = state.artifacts.find((a) => a.id === metadata.dataset_id);
  const counts = metadata.dataset_counts || dataset?.manifest.metadata.counts;
  const last = metadata.last_metrics || metadata.history?.at(-1) || {};
  const zero = Number(last["metrics/mAP50(B)"]) === 0;
  return `${counts ? `<p class="hint">Данные: ${counts.train} train · ${counts.val} val${counts.test ? ` · ${counts.test} test` : ""}</p>` : ""}${zero ? `<p class="callout warning">AP50 = 0: правильных обнаружений на val не получено.${counts?.train < 20 ? " Обучение на таком малом наборе не подтверждает качество модели." : ""}</p>` : ""}`;
}
export function advanced(body) {
  return `<details><summary>Дополнительные настройки</summary>${body}</details>`;
}
export function pathRow(path, action, label = "Выбрать") {
  return `<div class="path-row"><span class="path-value" title="${esc(path)}">${esc(path || "Не выбрано")}</span><button data-action="${action}">${label}</button></div>`;
}
export function versionSelector(kind, selected, id, label) {
  const items = state.artifacts.filter((a) => a.kind === kind);
  return field(
    label,
    `<select id="${id}"><option value="">Выберите результат</option>${items.map((a) => `<option value="${a.id}" ${selected === a.id ? "selected" : ""}>${esc(a.manifest.metadata.name || a.manifest.metadata.project?.dataset?.name || new Date(a.created).toLocaleString("ru"))} · ${a.id.slice(0, 6)}</option>`).join("")}</select>`,
  );
}
let popover;
export function wireHelp(root = document) {
  root.querySelectorAll("[data-help]").forEach((button) =>
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      closeHelp();
      const text = tips[button.dataset.help];
      if (!text) return;
      popover = document.createElement("div");
      popover.className = "popover";
      popover.role = "tooltip";
      popover.textContent = text;
      document.body.append(popover);
      const rect = button.getBoundingClientRect();
      const box = popover.getBoundingClientRect();
      popover.style.left =
        Math.max(10, Math.min(rect.left, innerWidth - box.width - 10)) + "px";
      popover.style.top =
        (rect.bottom + box.height + 15 < innerHeight
          ? rect.bottom + 8
          : Math.max(10, rect.top - box.height - 8)) + "px";
      button.setAttribute("aria-expanded", "true");
    }),
  );
}
function closeHelp() {
  popover?.remove();
  popover = null;
  document
    .querySelectorAll("[data-help][aria-expanded=true]")
    .forEach((e) => e.setAttribute("aria-expanded", "false"));
}
document.addEventListener("click", closeHelp);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeHelp();
});
window.addEventListener("scroll", closeHelp, true);
export async function choose({
  title = "Выберите папку",
  mode = "folder",
  extensions = [],
  multiple = false,
  start = state.root,
} = {}) {
  const dialog = document.querySelector("#file-picker");
  let current = start,
    selected = [];
  let resolve;
  const promise = new Promise((r) => (resolve = r));
  async function browse(path) {
    try {
      const data = await api("/fs/browse", {
        method: "POST",
        body: JSON.stringify({ path }),
      });
      current = data.path;
      selected = [];
      dialog.innerHTML = `<div class="dialog-head"><h2 id="picker-title">${title}</h2><button id="picker-close" aria-label="Закрыть">✕</button></div><div class="path-row"><button id="picker-up" aria-label="На уровень выше">↑</button><input id="picker-path" value="${esc(current)}"><button id="picker-open">Открыть</button></div><div class="files">${data.entries
        .filter(
          (e) =>
            e.directory ||
            (mode === "file" &&
              (!extensions.length || extensions.includes(e.suffix))),
        )
        .map(
          (e) =>
            `<button class="file-entry" data-path="${esc(e.path)}" data-directory="${e.directory}"><span>${e.directory ? "▸" : "◇"}</span><span>${esc(e.name)}</span><span>${e.directory ? "Папка" : Math.round(e.size / 1024) + " КБ"}</span></button>`,
        )
        .join(
          "",
        )}</div><p class="muted small-text" id="picker-selection">${mode === "folder" ? "Будет подключена открытая папка." : "Выберите файл" + (multiple ? "ы" : "") + "."}</p><div class="actions spaced"><button id="picker-cancel">Отмена</button><button class="primary" id="picker-select">${mode === "folder" ? "Выбрать эту папку" : "Выбрать"}</button></div>`;
      dialog.querySelector("#picker-close").onclick = () => finish(null);
      dialog.querySelector("#picker-cancel").onclick = () => finish(null);
      dialog.querySelector("#picker-up").onclick = () => browse(data.parent);
      dialog.querySelector("#picker-open").onclick = () =>
        browse(dialog.querySelector("#picker-path").value);
      dialog.querySelector("#picker-path").onkeydown = (e) => {
        if (e.key === "Enter") browse(e.target.value);
      };
      dialog.querySelectorAll(".file-entry").forEach(
        (button) =>
          (button.onclick = () => {
            if (button.dataset.directory === "true") {
              browse(button.dataset.path);
              return;
            }
            if (!multiple) selected = [];
            selected = selected.includes(button.dataset.path)
              ? selected.filter((p) => p !== button.dataset.path)
              : [...selected, button.dataset.path];
            dialog
              .querySelectorAll(".file-entry")
              .forEach((b) =>
                b.classList.toggle(
                  "selected",
                  selected.includes(b.dataset.path),
                ),
              );
            dialog.querySelector("#picker-selection").textContent =
              `Выбрано файлов: ${selected.length}`;
          }),
      );
      dialog.querySelector("#picker-select").onclick = async () => {
        const values = mode === "folder" ? [current] : selected;
        if (!values.length) return notice("Выберите файл", true);
        try {
          for (const path of values)
            await api("/fs/grant", {
              method: "POST",
              body: JSON.stringify({ path }),
            });
          finish(multiple ? values : values[0]);
        } catch (e) {
          notice(e.message, true);
        }
      };
    } catch (e) {
      notice(e.message, true);
    }
  }
  function finish(value) {
    dialog.close();
    resolve(value);
  }
  dialog.oncancel = () => resolve(null);
  await browse(current);
  dialog.showModal();
  return promise;
}
export function showResult(title, html) {
  const dialog = document.querySelector("#result-dialog");
  dialog.innerHTML = `<div class="dialog-head"><h2 id="result-title">${esc(title)}</h2><button id="result-close" aria-label="Закрыть">✕</button></div>${html}`;
  dialog.querySelector("#result-close").onclick = () => dialog.close();
  dialog.showModal();
  wireHelp(dialog);
  return dialog;
}
