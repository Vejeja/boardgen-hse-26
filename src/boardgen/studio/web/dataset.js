import { state, esc, notice, api, getPath } from "./api.js";
import {
  listPage,
  field,
  number,
  select,
  check,
  range,
  advanced,
  empty,
  choose,
  pathRow,
} from "./controls.js";
import { chosenFigures, splitCounts } from "./workflow_data.js";

function sourcesEditor(path, title) {
  const sources = getPath(state.project, path);
  return `<section class="source-row"><h3>${title}</h3><div class="actions"><button data-add-board="${path}">Добавить поле</button><button data-add-skybox="${path}">Добавить панораму</button></div>${sources.boards.map((b, i) => `<div class="board-row"><span class="small-text" title="${esc(b.image)}">${esc(b.image.split("/").at(-1))}</span>${field("Ширина, м", number(`${path}.boards.${i}.width_m`, b.width_m, 0.01, 10, 0.01))}${field("Длина, м", number(`${path}.boards.${i}.height_m`, b.height_m, 0.01, 10, 0.01))}<button class="small danger" data-remove-source="${path}.boards" data-index="${i}">Убрать</button></div>`).join("") || '<p class="hint">Поле не задано. Для подиума оно не требуется.</p>'}<h3>Панорамы World</h3>${sources.skyboxes.map((p, i) => `<div class="actions spaced"><span class="small-text" title="${esc(p)}">${esc(p.split("/").at(-1))}</span><button class="small danger" data-remove-source="${path}.skyboxes" data-index="${i}">Убрать</button></div>`).join("") || '<p class="hint">Без панорамы используется стандартный свет. Для разделения источников фона укажите панорамы явно.</p>'}</section>`;
}
function selectedModels() {
  const figures = chosenFigures(state.project);
  return `<div class="card"><div class="actions spaced"><h2>Модели для новых сцен</h2><button data-page="prepare" data-prepare-stage="compare">Изменить выбор</button></div><p class="hint">Закреплённые версии из «Сравнение и выбор». Эта страница задаёт размещение и размеры.</p>${
    figures.length
      ? `<table><thead><tr><th>Фигура</th><th>Закреплённая версия</th></tr></thead><tbody>${figures
          .map((f) => {
            const id = state.project.scenes.asset_versions[f.id];
            const a = state.artifacts.find((a) => a.id === id);
            return `<tr><td>${esc(f.name)}</td><td>${a ? `<button class="small" data-artifact="${id}">${esc(a.manifest.metadata.research_title || "3D-модель")} · ${id.slice(0, 8)}</button>` : '<span class="error-text">Версия не закреплена или недоступна</span>'}</td></tr>`;
          })
          .join("")}</tbody></table>`
      : empty("Сначала выберите модели для сцен.")
  }</div>`;
}
export function render() {
  const s = state.project.scenes,
    d = state.project.dataset;
  const counts = splitCounts(s.count, d.val_fraction, d.test_fraction);
  const versions = listPage(
    state.artifacts.filter((a) => ["pilot", "dataset"].includes(a.kind)),
    "datasets",
    5,
  );
  return `<div class="card"><h2>Состав и разделение данных</h2>${check("dataset.include_synthetic", d.include_synthetic, "Генерировать сцены Blender")}<p class="hint">Один генератор создаёт общую 3D-сцену: поверхность, модели с материалами, панораму World и перспективную камеру. Размеченные фотографии можно подключить отдельно.</p><div class="grid three">${field("Название версии", `<input data-bind="dataset.name" value="${esc(d.name)}">`)}${field("Доля val", number("dataset.val_fraction", d.val_fraction, 0.01, 0.99, 0.01))}${field("Доля test", number("dataset.test_fraction", d.test_fraction, 0, 0.99, 0.01))}</div><p class="hint">План синтетики: ${counts ? `${counts.train} train · ${counts.val} val · ${counts.test} test` : "Сумма долей val и test должна быть меньше 1"}. При test > 0 раздел сцены закрепляется до рендера.</p>${field(
    "Разделение источников фона",
    select("dataset.background_policy", d.background_policy, [
      ["none", "Общие фоны разрешены"],
      ["test_only", "Test отдельно; train и val могут иметь общие фоны"],
      ["all_splits", "Отдельные источники для каждого раздела"],
    ]),
  )}${advanced(
    field("Seed разделения", number("dataset.seed", d.seed, 0, 2147483647)) +
      field(
        "Похожие изображения между разделами",
        select("dataset.near_duplicates", d.near_duplicates, [
          ["review", "Проверить перед обучением"],
          ["error", "Остановить сборку"],
        ]),
      ),
  )}<h3>Размеченные фотографии</h3><button data-action="add-real">Добавить data.yaml</button>${d.real_sources
    .map(
      (source, i) =>
        `<div class="source-row"><div class="actions spaced"><span>${esc(source.path)}</span><button data-remove-real="${i}">Убрать</button></div>${field(
          "Назначение",
          select(`dataset.real_sources.${i}.role`, source.role, [
            ["development", "Train / val"],
            ["test_only", "Только test"],
          ]),
        )}${field("Группы одной съёмки", pathRow(source.groups_csv, `groups-${i}`))}${check(`dataset.real_sources.${i}.negative_labels_confirmed`, source.negative_labels_confirmed, "Кадры без разметки подтверждены как пустые")}</div>`,
    )
    .join("")}</div>${
    d.include_synthetic
      ? `${selectedModels()}<div class="card"><h2>Тип сцены и размещение</h2>${field(
          "Сцены",
          select("scenes.scene_type", s.scene_type, [
            ["board", "Игровое поле"],
            ["platform", "Одна фигура на подиуме"],
            ["mixed", "Поровну: поле и подиум"],
          ]),
        )}<p class="hint">Собственная подставка фигурки входит в её рамку; отдельный подиум считается фоном. Пустые сцены распределяются по типам при включённой стратификации.</p><div class="grid">${field("Фигур на поле", range("scenes.objects_per_scene", s.objects_per_scene, 1, 100))}${field("Доля пустых сцен", number("scenes.negative_fraction", s.negative_fraction, 0, 0.5, 0.01))}</div>${check("scenes.balanced_classes", s.balanced_classes, "Балансировать классы при размещении")}${check("scenes.stratified_negatives", s.stratified_negatives, "Распределять пустые сцены по разделам и типам")}${s.scene_type !== "board" ? `<h3>Одиночный подиум</h3><div class="grid">${field("Радиус, м", number("scenes.platform.radius_m", s.platform.radius_m, 0.001, 1, 0.001))}${field("Высота, м", number("scenes.platform.height_m", s.platform.height_m, 0.001, 1, 0.001))}${field("Угол камеры, °", range("scenes.platform.elevation_deg", s.platform.elevation_deg, 1, 90))}${field("Доля высоты кадра для фигуры", range("scenes.platform.frame_fraction", s.platform.frame_fraction, 0.1, 0.8, 0.01))}</div>` : ""}${advanced(
          field(
            "Относительные размеры",
            select("scenes.scale_basis", s.scale_basis, [
              ["nominal", "Базовая / измеренная высота"],
              ["shared_video", "По маскам одинаковой съёмки"],
            ]),
          ) +
            field(
              "Опорная фигура",
              select("scenes.reference_class", s.reference_class ?? "", [
                ["", "Первая выбранная"],
                ...chosenFigures(state.project).map((f) => [f.id, f.name]),
              ]),
            ) +
            field(
              "Базовая высота, м",
              number(
                "scenes.nominal_height_m",
                s.nominal_height_m,
                0.001,
                1,
                0.001,
              ),
            ) +
            chosenFigures(state.project)
              .map((f) =>
                field(
                  `${esc(f.name)}: измеренная высота, м`,
                  `<input type="number" min=".001" step=".001" data-height-figure="${f.id}" value="${s.height_overrides[f.id] ?? ""}" placeholder="По базовой высоте">`,
                ),
              )
              .join("") +
            check(
              "scenes.neutral_ablation",
              s.neutral_ablation,
              "Разрешить нейтральный материал для технических сцен",
            ),
        )}</div><div class="card"><h2>Поля и окружение по разделам</h2><p class="hint">Раздел наследует общие источники, пока не включён его отдельный набор. Ниже показаны источники, которые реально получит Blender; файлы производных фонов сохраняют общий источник.</p>${sourcesEditor("scenes", "Общие источники")}${[
          "train",
          "val",
          ...(d.test_fraction > 0 ? ["test"] : []),
        ]
          .map((split) => {
            const own = Object.hasOwn(s.split_sources, split),
              effective = own ? s.split_sources[split] : s;
            return `<section><label class="check-row"><input type="checkbox" data-own-sources="${split}" ${own ? "checked" : ""}>Отдельные источники ${split}</label>${own ? sourcesEditor(`scenes.split_sources.${split}`, split) : `<p class="hint">${split}: общие источники — ${effective.boards.length} полей, ${effective.skyboxes.length} панорам</p>`}</section>`;
          })
          .join(
            "",
          )}</div><div class="card"><h2>Камера и освещение</h2><div class="grid three">${field("Расстояние до центра поля, м", range("scenes.camera.distance_m", s.camera.distance_m, 0.1, 10, 0.05))}${field("Угол над полем, °", range("scenes.camera.elevation_deg", s.camera.elevation_deg, 1, 90))}${field("Азимут, °", range("scenes.camera.azimuth_deg", s.camera.azimuth_deg, -360, 360))}</div>${advanced(`<div class="grid">${field("FOV, °", number("scenes.camera.fov_deg", s.camera.fov_deg, 35, 65))}${field("Смещение XY, м", number("scenes.camera.jitter_xy_m", s.camera.jitter_xy_m, 0, 1, 0.005))}${field("Наклон горизонта, °", number("scenes.camera.jitter_roll_deg", s.camera.jitter_roll_deg, 0, 45))}${field("Экспозиция, EV", number("scenes.camera.exposure_ev", s.camera.exposure_ev, -10, 10, 0.1))}${field("Разброс экспозиции, EV", number("scenes.exposure_jitter_ev", s.exposure_jitter_ev, 0, 3, 0.1))}${field("Яркость World", number("scenes.environment_strength", s.environment_strength, 0, 20, 0.1))}${field("Поворот панорамы, °", range("scenes.environment_rotation_deg", s.environment_rotation_deg, -360, 360))}${field("Энергия света", range("scenes.light_energy", s.light_energy, 1, 10000))}</div>`)}</div><div class="card"><h2>Рендер</h2><div class="grid three">${field("Количество сцен", number("scenes.count", s.count, 2, 1000000))}${field("Разрешение, px", range("scenes.image_size", s.image_size, 64, 8192, 32, "", ["ширина", "высота"]))}${field("Сэмплы рендера", number("scenes.render_samples", s.render_samples, 1, 256))}</div><p class="hint">Больше сэмплов снижает шум и увеличивает время. Пробный рендер позволяет проверить кадры и рамки до сборки всего набора.</p>${advanced(field("Seed сцен", number("scenes.seed", s.seed, 0, 2147483647)) + field("Минимальная сторона рамки, px", number("scenes.min_side_px", s.min_side_px, 1, 512)) + field("Попыток размещения", number("scenes.max_attempts", s.max_attempts, 1, 100)))}</div>`
      : ""
  }<div class="sticky-actions actions">${d.include_synthetic ? '<button data-action="pilot">Пробный рендер — до 4 сцен</button>' : ""}<button class="primary" data-action="dataset">Собрать новую версию датасета</button></div><div class="card"><h2>Сохранённые датасеты</h2><p class="hint">Эти версии неизменяемы. Новый выбор моделей или настроек применяется только к следующей сборке.</p>${
    versions.rows
      .map(
        (a) =>
          `<div class="job-head"><div><strong>${a.kind === "pilot" ? "Пробный рендер" : esc(a.manifest.metadata.name || "Датасет")} · ${a.id.slice(0, 8)}</strong><p class="hint">${
            a.kind === "dataset"
              ? Object.entries(a.manifest.metadata.counts)
                  .map(([s, n]) => `${s}: ${n}`)
                  .join(" · ")
              : `${a.manifest.metadata.scenes.length} сцен`
          }</p></div><button data-artifact="${a.id}">Просмотреть</button></div>`,
      )
      .join("") || empty("Сохранённых версий пока нет.")
  }${versions.navigation}</div>`;
}
export function wire(redraw, run, showArtifact, onChange) {
  document.querySelector("[data-action=add-real]").onclick = async () => {
    const path = await choose({
      title: "Размеченный датасет: data.yaml",
      mode: "file",
      extensions: [".yaml", ".yml"],
    });
    if (!path) return;
    if (state.project.dataset.real_sources.some((s) => s.path === path))
      return notice("Этот датасет уже подключён");
    state.project.dataset.real_sources.push({
      path,
      role: "development",
      groups_csv: null,
      negative_labels_confirmed: false,
    });
    onChange(true);
  };
  document.querySelectorAll("[data-remove-real]").forEach(
    (b) =>
      (b.onclick = () => {
        state.project.dataset.real_sources.splice(
          Number(b.dataset.removeReal),
          1,
        );
        onChange(true);
      }),
  );
  document.querySelectorAll("[data-action^=groups-]").forEach(
    (b) =>
      (b.onclick = async () => {
        const path = await choose({
          title: "Группы съёмок: CSV",
          mode: "file",
          extensions: [".csv"],
        });
        if (path) {
          state.project.dataset.real_sources[
            Number(b.dataset.action.slice(7))
          ].groups_csv = path;
          onChange(true);
        }
      }),
  );
  document.querySelectorAll("[data-own-sources]").forEach(
    (b) =>
      (b.onchange = () => {
        if (b.checked)
          state.project.scenes.split_sources[b.dataset.ownSources] = {
            boards: structuredClone(state.project.scenes.boards),
            skyboxes: [...state.project.scenes.skyboxes],
          };
        else delete state.project.scenes.split_sources[b.dataset.ownSources];
        onChange(true);
      }),
  );
  document.querySelectorAll("[data-add-board], [data-add-skybox]").forEach(
    (b) =>
      (b.onclick = async () => {
        const board = Boolean(b.dataset.addBoard),
          path = b.dataset.addBoard || b.dataset.addSkybox;
        const files = await choose({
          title: board ? "Изображения поля" : "Панорамы World",
          mode: "file",
          multiple: true,
          extensions: board
            ? [".png", ".jpg", ".jpeg", ".webp"]
            : [".hdr", ".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff"],
        });
        if (!files) return;
        const sources = getPath(state.project, path);
        for (const image of files) {
          if (board && !sources.boards.some((b) => b.image === image)) {
            const info = await api("/image-info", {
              method: "POST",
              body: JSON.stringify({ path: image }),
            });
            sources.boards.push({
              image,
              width_m: 0.5,
              height_m: (0.5 * info.height) / info.width,
            });
          } else if (!board && !sources.skyboxes.includes(image))
            sources.skyboxes.push(image);
        }
        onChange(true);
      }),
  );
  document.querySelectorAll("[data-remove-source]").forEach(
    (b) =>
      (b.onclick = () => {
        getPath(state.project, b.dataset.removeSource).splice(
          Number(b.dataset.index),
          1,
        );
        onChange(true);
      }),
  );
  document
    .querySelector("[data-action=pilot]")
    ?.addEventListener("click", () => run("pilot"));
  document.querySelector("[data-action=dataset]").onclick = () =>
    run("dataset");
  document
    .querySelectorAll("[data-artifact]")
    .forEach((b) => (b.onclick = () => showArtifact(b.dataset.artifact)));
}
