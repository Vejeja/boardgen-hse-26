import { editConfiguration } from "./configuration.js";
import {
  renderComparison,
  wireComparison,
  disposeComparison,
} from "./comparison_page.js";
export const dispose = disposeComparison;
import {
  state,
  esc,
  recipe,
  scopedFigures,
  artifactUrl,
  api,
  notice,
  merged,
} from "./api.js";
import { connectVideoFolder } from "./video_inputs.js";
import {
  listPage,
  field,
  input,
  number,
  select,
  check,
  advanced,
  help,
  empty,
  choose,
} from "./controls.js";
const stages = [
  ["video", "Видео"],
  ["matte", "Убрать фон"],
  ["shape", "3D-форма"],
  ["texture", "Текстуры"],
  ["results", "Результаты"],
  ["compare", "Сравнение и выбор"],
];
export function currentFrames(figure) {
  const r = merged(
    state.project.prepare.defaults,
    state.project.prepare.overrides[figure.id],
  );
  return state.artifacts.find(
    (a) =>
      a.kind === "frames" &&
      a.figure_id === figure.id &&
      JSON.stringify(a.manifest.metadata.clip) ===
        JSON.stringify(figure.clip) &&
      JSON.stringify(a.manifest.metadata.recipe.matte) ===
        JSON.stringify(r.matte) &&
      a.manifest.metadata.recipe.candidate_frames ===
        r.shape.candidate_frames &&
      a.manifest.metadata.recipe.frame_size === r.shape.frame_size,
  );
}
function videoStage() {
  const page = listPage(scopedFigures(), "videos", 5);
  return `<div class="actions spaced"><h2>Видео и названия фигур</h2><button data-action="connect-more">Выбрать папку</button></div><div class="video-list">${
    page.rows
      .map((f) => {
        const info = state.videos.find((v) => v.id === f.id);
        return `<div class="video-row"><div class="video-header"><label class="field" style="margin:0"><span class="label">Название фигуры</span><input data-figure="${f.id}" data-figure-field="name" value="${esc(f.name)}"></label><button class="small danger" data-remove="${f.id}">Убрать видео</button></div><p class="small-text muted">${esc(f.video.split("/").at(-1))} · ${info?.duration_s ? info.duration_s.toFixed(1) + " с" : ""} · ${info?.width || "?"} × ${info?.height || "?"}</p><div class="clip"><label class="field"><span class="label">Начало, с ${help("clip")}</span><input type="number" min="0" step=".1" data-figure="${f.id}" data-figure-field="clip.start_s" value="${f.clip.start_s}"></label><label class="field"><span class="label">Конец, с</span><input type="number" min="0" step=".1" data-figure="${f.id}" data-figure-field="clip.end_s" value="${f.clip.end_s ?? ""}" placeholder="До конца"></label><button class="small" data-full-clip="${f.id}">Всё видео</button></div><details><summary>Посмотреть видео</summary><video controls preload="none" src="/api/video/${f.id}"></video></details></div>`;
      })
      .join("") || empty("Подключите папку с видео в разделе «Проект».")
  }</div>${page.navigation}`;
}
function matteStage() {
  const r = recipe();
  return `<h2>Выделение фигуры</h2><div class="grid">${field(
    "Метод маскирования",
    select("prep.matte.method", r.matte.method, [
      ["birefnet", "BiRefNet — нейросетевая маска"],
      ["classic", "Статичный фон — классическая маска"],
    ]),
    "matte",
  )}${field("Количество кадров", number("prep.shape.candidate_frames", r.shape.candidate_frames, 10, 400, 5), "frames")}</div>${advanced(
    `<div class="grid">${
      r.matte.method === "birefnet"
        ? field(
            "Вход BiRefNet, px",
            select(
              "prep.matte.input_size",
              r.matte.input_size,
              [
                [512, "512"],
                [1024, "1024"],
                [2048, "2048"],
              ].map(([v, n]) => [String(v), n]),
            ),
            "maskSize",
          )
        : ""
    }${field("Размер подготовленного кадра, px", number("prep.shape.frame_size", r.shape.frame_size, 256, 2048, 32))}</div>${r.matte.method === "birefnet" ? check("prep.matte.cross_check", r.matte.cross_check, "Проверить маски вторым методом", "crossCheck") : ""}`,
  )}<div class="callout">После подготовки кадров можно проверить границы фигуры и выбрать ракурсы для 3D-модели.</div>`;
}
function shapeStage() {
  const r = recipe();
  let figures = scopedFigures();
  let f =
    figures.find((f) => f.id === state.selectedFramesFigure) || figures[0];
  const frames = f && currentFrames(f);
  return `<h2>Восстановление 3D-формы</h2>${field(
    "Метод реконструкции",
    select("prep.shape.method", r.shape.method, [
      ["mv", "Hunyuan3D MV Turbo — четыре ракурса"],
      ["spirula", "Spirula — серия кадров и камеры"],
    ]),
  )}${
    r.shape.method === "mv"
      ? `${advanced(
          field(
            "Разрешение поверхности",
            select(
              "prep.shape.mesh_resolution",
              String(r.shape.mesh_resolution),
              [
                ["128", "128"],
                ["256", "256"],
                ["384", "384"],
              ],
            ),
            "mesh",
          ) +
            field(
              "Seed генерации",
              number("prep.shape.seed", r.shape.seed, 0, 2147483647),
              "seed",
            ) +
            field(
              "Лимит расчёта модели, с",
              number(
                "prep.shape.timeout_seconds",
                r.shape.timeout_seconds,
                60,
                86400,
              ),
            ),
        )}<h3 style="margin-top:24px">Четыре стороны фигуры ${help("views")}</h3>${figures.length > 1 ? `<label class="field"><span class="label">Фигура для выбора ракурсов</span><select id="frame-figure">${figures.map((x) => `<option value="${x.id}" ${x.id === f.id ? "selected" : ""}>${esc(x.name)}</option>`).join("")}</select></label>` : ""}${
          f && frames
            ? `<div class="view-selection">${[
                ["front", "Передняя сторона"],
                ["left", "Левая сторона"],
                ["back", "Тыльная сторона"],
                ["right", "Правая сторона"],
              ]
                .map(
                  ([tag, title]) =>
                    `<button data-view-tag="${tag}" class="${state.viewTag === tag ? "active" : ""}"><strong>${title}</strong>${f.reviewed_views[tag] !== undefined ? `<img src="${artifactUrl(frames, `views/train/images/view_${String(f.reviewed_views[tag]).padStart(4, "0")}.png`)}"><span class="small-text">Кадр ${f.reviewed_views[tag]}</span>` : '<p class="small-text muted">Выберите кадр ниже</p>'}</button>`,
                )
                .join(
                  "",
                )}</div><p class="hint">Нажмите кадр для стороны «${{ front: "Передняя сторона", left: "Левая сторона", back: "Тыльная сторона", right: "Правая сторона" }[state.viewTag]}». Кадры проверки скрыты.</p><div class="frames">${frames.manifest.metadata.fit.map((i) => `<button class="frame ${Object.values(f.reviewed_views).includes(i) ? "selected" : ""}" data-view-index="${i}" data-view-figure="${f.id}" data-frames-id="${frames.id}" aria-label="Кадр ${i}"><img loading="lazy" src="${artifactUrl(frames, `views/train/images/view_${String(i).padStart(4, "0")}.png`)}"><span>Кадр ${i}${Object.entries(f.reviewed_views).find(([t, v]) => v === i) ? " · " + { front: "передняя сторона", left: "левая сторона", back: "тыльная сторона", right: "правая сторона" }[Object.entries(f.reviewed_views).find(([t, v]) => v === i)[0]] : ""}</span></button>`).join("")}</div>`
            : empty(
                "Сначала подготовьте кадры с масками. Они появятся здесь после завершения задания.",
              )
        }`
      : `<div class="grid">${field(
          "Положение камеры",
          select("prep.shape.cameras", r.shape.cameras, [
            ["orbit", "Orbit — вращение фигуры"],
            ["sfm", "SfM / SIFT — по совпадениям"],
          ]),
          "cameras",
        )}${field(
          "Поверхность",
          select("prep.shape.extraction", r.shape.extraction || "native", [
            ["native", "Native"],
            ["depth_fusion", "Depth fusion"],
          ]),
        )}${field("Шаги обучения Spirula", number("prep.shape.iterations", r.shape.iterations, 1000, 100000, 500))}</div>${advanced(
          `<div class="grid">${field(
            "Профиль поверхности",
            select("prep.shape.profile", r.shape.profile, [
              ["control", "Control"],
              ["detail", "Detail — тонкие детали"],
            ]),
          )}${field("Лимит Gaussian-примитивов", number("prep.shape.cap", r.shape.cap, 10000, 1000000, 10000))}${field("Лимит расчёта модели, с", number("prep.shape.timeout_seconds", r.shape.timeout_seconds, 60, 86400))}</div>`,
        )}`
  }${(r.shape.method === "spirula" && r.shape.cameras === "orbit") || r.appearance.registration === "orbit" ? `<details open><summary>Начальные параметры оборота ${help("period")}</summary>${figures.map((f) => `<h3>${esc(f.name)}</h3><div class="grid three">${field("Передний кадр", `<input type="number" min="0" data-figure="${f.id}" data-figure-field="front_view" value="${f.front_view ?? ""}">`)}${field("Кадров на оборот", `<input type="number" min="5" data-figure="${f.id}" data-figure-field="period_views" value="${f.period_views ?? ""}">`, "period")}${field("Фокус, px", `<input type="number" min="1" data-figure="${f.id}" data-figure-field="focal_px" value="${f.focal_px ?? ""}">`, "focal")}</div>`).join("")}</details>` : ""}`;
}
function textureStage() {
  const r = recipe();
  const opts = [
    ["photo", "Фототекстура из видео"],
    ...(r.shape.method === "spirula"
      ? [["native", "Текстура реконструкции (Spirula)"]]
      : []),
    ["neutral", "Нейтральный материал — для сравнения"],
  ];
  return `<h2>Цвет и детали поверхности</h2>${field("Текстурирование", select("prep.appearance.method", r.appearance.method, opts), "texture")}${
    r.appearance.method === "photo"
      ? advanced(
          `<div class="grid">${
            r.shape.method === "mv"
              ? field(
                  "Регистрация фотографий",
                  select(
                    "prep.appearance.registration",
                    r.appearance.registration,
                    [
                      ["silhouette", "Подгонка по силуэтам"],
                      ["orbit", "Общая регистрация к orbit"],
                    ],
                  ),
                  "registration",
                )
              : ""
          }${field(
            "Размер карты текстуры",
            select(
              "prep.appearance.texture_size",
              String(r.appearance.texture_size),
              [
                ["1024", "1024 × 1024"],
                ["2048", "2048 × 2048"],
                ["4096", "4096 × 4096"],
              ],
            ),
            "atlas",
          )}${
            r.shape.method === "mv"
              ? field(
                  "Фоторакурсов для текстуры",
                  select(
                    "prep.appearance.photo_views",
                    String(r.appearance.photo_views),
                    [
                      ["4", "4 — выбранные стороны"],
                      ["12", "12 — по одному обороту"],
                      ["24", "24 — по одному обороту"],
                    ],
                  ),
                  "photoViews",
                )
              : ""
          }</div>`,
        )
      : r.appearance.method === "native"
        ? advanced(
            field(
              "Размер карты текстуры",
              select(
                "prep.appearance.texture_size",
                String(r.appearance.texture_size),
                [
                  ["1024", "1024 × 1024"],
                  ["2048", "2048 × 2048"],
                  ["4096", "4096 × 4096"],
                ],
              ),
              "atlas",
            ),
          )
        : '<div class="callout warning">Нейтральный материал показывает геометрию. Для цветного датасета используйте текстуру.</div>'
  }`;
}
function resultsStage() {
  const ids = scopedFigures().map((f) => f.id);
  const items = state.artifacts.filter(
    (a) => ["frames", "figure"].includes(a.kind) && ids.includes(a.figure_id),
  );
  const page = listPage(items, "prepared", 6);
  return `<div class="actions spaced"><h2 style="margin:0">История подготовки</h2><button data-stage="compare">Сравнить и выбрать</button></div><p class="hint">Здесь хранятся готовые версии и их параметры. Выбор для новых сцен выполняется в «Сравнение и выбор».</p>${
    items.length
      ? `<div class="artifacts">${page.rows
          .map((a) => {
            const selected =
              a.kind === "figure" &&
              state.project.scenes.asset_versions[a.figure_id] === a.id &&
              (state.project.scenes.selected_classes === null ||
                state.project.scenes.selected_classes.includes(a.figure_id));
            return `<article class="artifact"><h3>${esc(state.project.figures.find((f) => f.id === a.figure_id)?.name)}</h3><span class="badge">${a.kind === "frames" ? "Кадры и маски" : "3D-модель"}</span>${selected ? '<span class="badge">Выбрана для новых сцен</span>' : ""}${a.kind === "frames" ? `<img loading="lazy" src="${artifactUrl(a, a.manifest.metadata.contact)}">` : ""}<p class="hint">${new Date(a.created).toLocaleString("ru")} · ${a.id.slice(0, 8)}</p><p class="muted">${esc(a.manifest.metadata.research_title || a.manifest.metadata.recipe?.shape.method || "")}</p><div class="actions"><button class="small" data-artifact="${a.id}">Посмотреть</button><button class="small" data-recipe="${a.id}">Параметры и новый рецепт</button></div></article>`;
          })
          .join("")}</div>`
      : empty("Подготовьте кадры или подключите сохранённые модели.")
  }${page.navigation}`;
}
function scopeMarkup() {
  if (state.stage === "compare") return "";
  const history = state.stage === "results";
  const individual = state.scope !== "all";
  return `<div class="scope"><label for="prepare-scope">${history ? "Фигура в истории" : `Настройки обработки ${help("scope")}`}</label><select id="prepare-scope"><option value="all" ${individual ? "" : "selected"}>Все фигуры</option>${state.project.figures.map((f) => `<option value="${f.id}" ${state.scope === f.id ? "selected" : ""}>${esc(f.name)}</option>`).join("")}</select><span class="badge">Фигур в проекте: ${state.project.figures.length}</span>${individual && !history ? '<button class="small" data-action="reset-overrides">Сбросить исключения</button>' : ""}</div>${individual && !history ? '<p class="hint">Изменённые поля действуют только для выбранной фигуры. Остальные наследуются от общих настроек.</p>' : ""}`;
}
export function render() {
  const count = state.project.figures.length;
  const content = {
    video: videoStage,
    matte: matteStage,
    shape: shapeStage,
    texture: textureStage,
    results: resultsStage,
    compare: renderComparison,
  }[state.stage]();
  const actions = ["results", "compare"].includes(state.stage)
    ? '<div class="actions"><button data-page="dataset">Перейти к сценам и датасету</button><button data-stage="shape">Изменить настройки подготовки</button></div>'
    : `<div class="sticky-actions"><div class="stage-label"><span>Кадры и маски</span> → <span>3D-форма</span> → <span>Текстура</span></div><div class="actions"><button data-action="frames" ${!count ? "disabled" : ""}>Подготовить кадры</button><button class="primary" data-action="prepare" ${!count ? "disabled" : ""}>Подготовить по текущему методу</button><button data-action="prepare-candidates" ${!count ? "disabled" : ""}>Подготовить три варианта</button>${help("views")}</div><p class="hint">Вкладки выше задают общий рецепт. «Подготовить» выполняет кадры → камеры → форму → текстуру последовательно; результат появится в истории.</p></div>`;
  return `<div class="card">${scopeMarkup()}<div class="tabs" role="tablist">${stages.map(([key, label]) => `<button role="tab" aria-selected="${state.stage === key}" class="${state.stage === key ? "active" : ""}" data-stage="${key}">${label}</button>`).join("")}</div><div id="prepare-stage">${content}</div></div>${actions}`;
}
export function wire(redraw, run, showArtifact, onChange, flush) {
  if (state.stage === "compare") wireComparison(redraw, flush, showArtifact);
  document
    .querySelector("[data-action=prepare-candidates]")
    ?.addEventListener("click", () =>
      run("prepare-candidates", {
        figure_ids: scopedFigures().map((f) => f.id),
      }),
    );
  document.querySelector("#prepare-scope")?.addEventListener("change", (e) => {
    state.scope = e.target.value;
    state.selectedFramesFigure = null;
    state.listPages.videos = 0;
    state.listPages.prepared = 0;
    redraw();
  });
  document.querySelectorAll("[data-stage]").forEach(
    (b) =>
      (b.onclick = () => {
        state.stage = b.dataset.stage;
        redraw();
      }),
  );
  document
    .querySelector("[data-action=frames]")
    ?.addEventListener("click", () =>
      run("frames", { figure_ids: scopedFigures().map((f) => f.id) }),
    );
  document
    .querySelector("[data-action=prepare]")
    ?.addEventListener("click", () =>
      run("prepare", { figure_ids: scopedFigures().map((f) => f.id) }),
    );
  document
    .querySelector("[data-action=reset-overrides]")
    ?.addEventListener("click", () => {
      delete state.project.prepare.overrides[state.scope];
      onChange(true);
    });
  document
    .querySelector("[data-action=connect-more]")
    ?.addEventListener("click", () => connectVideoFolder(flush, redraw));
  document.querySelectorAll("[data-remove]").forEach(
    (b) =>
      (b.onclick = () => {
        const id = b.dataset.remove;
        state.project.figures = state.project.figures.filter(
          (f) => f.id !== id,
        );
        delete state.project.prepare.overrides[id];
        delete state.project.scenes.height_overrides[id];
        delete state.project.scenes.asset_versions[id];
        if (state.project.scenes.selected_classes !== null)
          state.project.scenes.selected_classes =
            state.project.scenes.selected_classes.filter((i) => i !== id);
        if (state.project.scenes.reference_class === id) {
          state.project.scenes.reference_class = null;
          state.project.scenes.scale_basis = "nominal";
        }
        if (state.scope === id) state.scope = "all";
        state.datasetId = null;
        state.modelId = null;
        state.selectedFramesFigure = null;
        onChange(true);
      }),
  );
  document.querySelectorAll("[data-full-clip]").forEach(
    (b) =>
      (b.onclick = () => {
        const f = state.project.figures.find(
          (f) => f.id === b.dataset.fullClip,
        );
        f.clip = { start_s: 0, end_s: null };
        f.reviewed_views = {};
        f.reviewed_frames_id = null;
        onChange(true);
      }),
  );
  document.querySelector("#frame-figure")?.addEventListener("change", (e) => {
    state.selectedFramesFigure = e.target.value;
    redraw();
  });
  document.querySelectorAll("[data-view-tag]").forEach(
    (b) =>
      (b.onclick = () => {
        state.viewTag = b.dataset.viewTag;
        redraw();
      }),
  );
  document.querySelectorAll("[data-view-index]").forEach(
    (b) =>
      (b.onclick = () => {
        const f = state.project.figures.find(
          (f) => f.id === b.dataset.viewFigure,
        );
        const index = Number(b.dataset.viewIndex);
        if (
          Object.entries(f.reviewed_views).some(
            ([t, v]) => v === index && t !== state.viewTag,
          )
        )
          return notice("Этот кадр уже выбран для другой стороны", true);
        f.reviewed_views[state.viewTag] = index;
        f.reviewed_frames_id = b.dataset.framesId;
        const next = ["front", "left", "back", "right"].find(
          (t) => f.reviewed_views[t] === undefined,
        );
        if (next) state.viewTag = next;
        onChange(true);
      }),
  );
  document.querySelectorAll("[data-recipe]").forEach(
    (b) =>
      (b.onclick = async () => {
        try {
          const a = await api(`/artifacts/${b.dataset.recipe}`);
          await editConfiguration(redraw, flush, a);
        } catch (error) {
          notice(error.message, true);
        }
      }),
  );
  document
    .querySelectorAll("[data-artifact]")
    .forEach((b) => (b.onclick = () => showArtifact(b.dataset.artifact)));
}
