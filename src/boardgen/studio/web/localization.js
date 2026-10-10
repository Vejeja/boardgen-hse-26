import { state, api, esc, notice } from "./api.js";
import {
  field,
  number,
  select,
  check,
  pathRow,
  choose,
  advanced,
} from "./controls.js";

export function renderLocalization() {
  const l = state.project.localization;
  const c = l.camera;
  return `${field("Фото, видео или папка изображений", pathRow(state.predictionSource, "prediction-source"))}<button class="small" data-action="prediction-folder">Выбрать папку</button>
    <h3>Камера</h3><p class="hint">Введите параметры для исходного разрешения калибровки. Начальные значения — пример Blender: 1152 × 1152, FOV 45°. При пропорциональном изменении разрешения K пересчитывается автоматически.</p>
    <div class="grid">${[
      ["width", "Ширина калибровки, px", 16, 32768, 1, "calibrationSize"],
      ["height", "Высота калибровки, px", 16, 32768, 1, "calibrationSize"],
      ["fx", "Фокус fx, px", 1, 100000, 0.001, "calibrationFocal"],
      ["fy", "Фокус fy, px", 1, 100000, 0.001, "calibrationFocal"],
      ["cx", "Главная точка cx, px", 0, 32768, 0.001, "calibrationCenter"],
      ["cy", "Главная точка cy, px", 0, 32768, 0.001, "calibrationCenter"],
    ]
      .map(([key, title, min, max, step, help]) =>
        field(
          title,
          number(`localization.camera.${key}`, c[key], min, max, step),
          help,
        ),
      )
      .join("")}</div>
    ${advanced(`<div class="grid">${c.distortion.map((v, i) => field(["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"][i], number(`localization.camera.distortion.${i}`, v, -100, 100, 0.00001), "calibrationDistortion")).join("")}</div>`)}
    <h3>Игровое поле</h3>${field("Исходное изображение карты", pathRow(l.board.image, "localization-board"), "localizationBoard")}
    <div class="grid">${field("Ширина поля, м", number("localization.board.width_m", l.board.width_m, 0.01, 100, 0.001), "localizationBoard")}${field("Длина поля, м", number("localization.board.height_m", l.board.height_m, 0.01, 100, 0.001), "localizationBoard")}${field(
      "Определение поля",
      select("localization.registration", l.registration, [
        ["template", "По совпадениям с картой"],
        ["corners", "Четыре угла вручную"],
      ]),
      "localizationRegistration",
    )}</div>
    ${check("localization.fixed_camera", l.fixed_camera, "Камера неподвижна в пределах одного видео", "localizationFixed")}
    ${l.registration === "corners" ? `<p class="hint">Укажите на кадре углы в порядке исходной карты: верхний левый → верхний правый → нижний правый → нижний левый. Ручные углы подходят для одного фото или неподвижного видео.</p><button data-action="localization-preview">Открыть первый кадр и указать углы</button><p id="corner-count" class="hint">Сохранено углов: ${l.corners.length} / 4</p><div id="corner-editor"></div>` : ""}
    ${advanced(`<div class="grid">${field("Каждый N-й кадр видео", number("localization.video_stride", l.video_stride, 1, 300), "localizationStride")}${field("Кадров в галерее", number("localization.preview_limit", l.preview_limit, 1, 500), "localizationPreview")}</div>`)}
    <p class="hint">Используются GLB и размеры из датасета обучения выбранного YOLO. Положения оцениваются по полным рамкам; перекрытия могут давать ошибку. Поворот модели приблизительный. Все обработанные координаты сохраняются, независимо от лимита галереи.</p>
    <button class="primary" data-action="localize" ${!state.modelId || !state.predictionSource ? "disabled" : ""}>Восстановить расстановку фигур</button>`;
}

export function wireLocalization(redraw, run, onChange) {
  document
    .querySelector('[data-action="localize"]')
    ?.addEventListener("click", () =>
      run("localize", {
        model_id: state.modelId,
        source: state.predictionSource,
      }),
    );
  document
    .querySelector('[data-action="localization-board"]')
    ?.addEventListener("click", async () => {
      const path = await choose({
        title: "Исходное изображение игрового поля",
        mode: "file",
        extensions: [".png", ".jpg", ".jpeg", ".webp"],
      });
      if (path) {
        state.project.localization.board.image = path;
        state.project.localization.corners = [];
        onChange(true);
      }
    });
  document
    .querySelector('[data-action="localization-preview"]')
    ?.addEventListener("click", async () => {
      try {
        const preview = await api("/localization-preview", {
          method: "POST",
          body: JSON.stringify({ path: state.predictionSource }),
        });
        const root = document.querySelector("#corner-editor");
        if (!root) return;
        let points = [];
        root.innerHTML = `<div style="position:relative;max-width:900px;margin-top:12px"><img id="corner-image" alt="Кадр для выбора углов поля" src="${esc(preview.image)}" style="width:100%;display:block;cursor:crosshair"><svg id="corner-markers" viewBox="0 0 1 1" preserveAspectRatio="none" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none"></svg></div><button class="small" id="corner-reset">Начать выбор заново</button>`;
        const count = document.querySelector("#corner-count");
        const reset = () => {
          points = [];
          state.project.localization.corners = [];
          root.querySelector("#corner-markers").innerHTML = "";
          count.textContent = "Выбрано углов: 0 / 4";
          onChange();
        };
        root.querySelector("#corner-reset").onclick = reset;
        root.querySelector("#corner-image").onclick = (event) => {
          if (points.length === 4) return;
          const rect = event.target.getBoundingClientRect();
          points.push([
            (event.clientX - rect.left) / rect.width,
            (event.clientY - rect.top) / rect.height,
          ]);
          root.querySelector("#corner-markers").innerHTML = points
            .map(
              ([x, y], i) =>
                `<circle cx="${x}" cy="${y}" r=".008" fill="#ffe63d"/><text x="${x + 0.012}" y="${y}" font-size=".025" fill="#ffe63d">${i + 1}</text>`,
            )
            .join("");
          count.textContent = `Выбрано углов: ${points.length} / 4`;
          if (points.length === 4) {
            state.project.localization.corners = points;
            onChange();
          }
        };
        reset();
      } catch (error) {
        notice(error.message, true);
      }
    });
}
