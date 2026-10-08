import { state, esc, api, artifactUrl, notice } from "./api.js";
import {
  showResult,
  empty,
  listPage,
  wirePages,
  trainingContext,
} from "./controls.js";
import { modelViewer } from "./viewer.js";
import { imagePrediction } from "./result_data.js";
function annotated(
  a,
  path,
  rows,
  names,
  caption = "",
  dashed = false,
  color = "#44e3a3",
) {
  return `<figure><div style="position:relative"><img loading="lazy" src="${artifactUrl(a, path)}"><svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Рамки разметки" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none">${rows.map((r) => `<rect x="${r[1] - r[3] / 2}" y="${r[2] - r[4] / 2}" width="${r[3]}" height="${r[4]}" fill="none" stroke="${color}" ${dashed ? 'stroke-dasharray=".018 .01"' : ""} stroke-width=".004"/>`).join("")}</svg></div><figcaption>${esc(caption)}${rows.length ? "<br>" + rows.map((r) => esc(names[r[0]] ?? r[0])).join(", ") : " · без фигур"}</figcaption></figure>`;
}
function plot(history) {
  const values = history
    .map((r) => Number(r["metrics/mAP50-95(B)"] ?? r["metrics/mAP50-95"]))
    .filter(Number.isFinite);
  if (!values.length) return "";
  const points = values
    .map(
      (v, i) =>
        `${30 + (i / Math.max(1, values.length - 1)) * 650},${170 - v * 140}`,
    )
    .join(" ");
  return `<p class="hint">mAP50–95 по эпохам</p><svg class="plot" viewBox="0 0 710 200" role="img" aria-label="mAP50–95 по эпохам"><path d="M30 25V170H690" fill="none" stroke="#9fac9f"/><polyline points="${points}" fill="none" stroke="#276d50" stroke-width="3"/><text x="8" y="33" font-size="11">1</text><text x="8" y="172" font-size="11">0</text><text x="30" y="190" font-size="11">1</text><text x="650" y="190" font-size="11">${values.length} эпох</text></svg>`;
}
export async function showArtifact(id) {
  if (!state.artifacts.some((x) => x.id === id)) return;
  const a = await api(`/artifacts/${id}`);
  const m = a.manifest.metadata;
  const files = Object.keys(a.manifest.files);
  let html = "";
  let title = {
    figure: "3D-модель",
    frames: "Кадры и маски",
    pilot: "Пробный рендер",
    dataset: "Версия датасета",
    train: "Обучение YOLO",
    evaluate: "Оценка качества",
    predict: "Предсказания",
  }[a.kind];
  if (a.kind === "figure")
    html = `<p>${esc(m.name)} · ${m.faces.toLocaleString("ru")} граней · ${m.origin === "research-import" ? "прежние вычисления" : "результат обработки"}</p><div class="viewer" id="model-viewer"></div><label class="check-row"><input type="checkbox" id="neutral-model">Показать геометрию без текстуры</label><div class="actions"><a href="${artifactUrl(a, m.mesh)}" download>Скачать GLB</a><a href="${artifactUrl(a, "manifest.json")}" target="_blank">Происхождение и параметры</a></div>${m.research_note ? `<p class="hint">${esc(m.research_note)}</p>` : ""}`;
  if (a.kind === "frames")
    html = `<img style="max-width:100%" src="${artifactUrl(a, m.contact)}"><p class="hint">${m.fit.length} кадров для построения · ${m.heldout.length} для проверки</p><table><thead><tr><th>Кадр</th><th>Резкость</th><th>Граница кадра</th><th>Согласие масок (IoU)</th></tr></thead><tbody>${m.diagnostics.map((d) => `<tr><td>${d.view}</td><td>${d.sharpness.toFixed(1)}</td><td>${d.touches_border ? "Касается" : "Свободна"}</td><td>${d.cross_check_iou?.toFixed(3) ?? "—"}</td></tr>`).join("")}</tbody></table>`;
  if (a.kind === "pilot") {
    const records = await (
      await fetch(artifactUrl(a, "render/scenes.json"))
    ).json();
    html = `<div class="gallery">${records
      .map((r) =>
        annotated(
          a,
          r.image.replace(a.directory + "/", ""),
          r.labels,
          m.project.figures.map((f) => f.name),
          `Seed ${r.seed}`,
        ),
      )
      .join("")}</div>`;
  }
  if (a.kind === "dataset") {
    const data = await (await fetch(artifactUrl(a, "dataset.json"))).json();
    const records = data.records;
    html = `<div class="metrics">${Object.entries(m.counts)
      .map(
        ([s, n]) =>
          `<div class="metric"><span>${s}</span><strong>${n}</strong></div>`,
      )
      .join(
        "",
      )}</div><p class="hint">Классы: ${m.names.map(esc).join(", ")}</p>${m.warnings.map((w) => `<p class="callout warning">${esc(w)}</p>`).join("")}<div class="actions"><a href="${artifactUrl(a, "data.yaml")}" download>data.yaml</a><a href="${artifactUrl(a, "dataset.json")}" target="_blank">Проверка разделения и происхождение</a></div><div class="tabs">${[
      "train",
      "val",
      "test",
      "near",
    ]
      .filter((s) => (s === "near" ? data.near_duplicate_count : m.counts[s]))
      .map(
        (s) =>
          `<button data-gallery-split="${s}">${s === "near" ? "Похожие пары (" + data.near_duplicate_count + ")" : s}</button>`,
      )
      .join(
        "",
      )}</div><div class="gallery" id="dataset-gallery"></div><div id="dataset-pages"></div>`;
    const dialog = showResult(title, html);
    function gallery(split) {
      let rows = records.filter((r) => r.split === split);
      if (split === "near") {
        const names = new Set(data.near_duplicates.flatMap((p) => [p.a, p.b]));
        rows = records.filter((r) => names.has(r.name));
      }
      const page = listPage(rows, `gallery-${a.id}-${split}`, 12);
      dialog.querySelector("#dataset-gallery").innerHTML = page.rows
        .map((r) =>
          annotated(a, r.image, r.labels, m.names, `${r.split} · ${r.group}`),
        )
        .join("");
      dialog
        .querySelectorAll("[data-gallery-split]")
        .forEach((b) =>
          b.classList.toggle("active", b.dataset.gallerySplit === split),
        );
      dialog.querySelector("#dataset-pages").innerHTML = page.navigation;
      wirePages(dialog.querySelector("#dataset-pages"), () => gallery(split));
    }
    dialog
      .querySelectorAll("[data-gallery-split]")
      .forEach((b) => (b.onclick = () => gallery(b.dataset.gallerySplit)));
    gallery("train");
    return;
  }
  if (a.kind === "train")
    html = `<p class="hint">Датасет ${esc(m.dataset_id.slice(0, 6))} · ${esc(m.device)} · ${m.history.length} эпох</p>${trainingContext(m)}${plot(m.history)}<div class="actions"><a href="${artifactUrl(a, m.best)}" download>best.pt</a><a href="${artifactUrl(a, m.last)}" download>last.pt</a><a href="${artifactUrl(a, "training.json")}" target="_blank">Параметры и метрики</a></div>${files
      .filter((p) => p.endsWith("results.png"))
      .map(
        (p) =>
          `<img style="width:100%;margin-top:20px" src="${artifactUrl(a, p)}">`,
      )
      .join("")}`;
  if (a.kind === "evaluate") {
    const dataset = state.artifacts.find((x) => x.id === m.dataset_id);
    const names =
      dataset?.manifest.metadata.names || m.per_class.map((c) => c.name);
    const ap = Object.entries(m.metrics).filter(([k]) => k.includes("mAP"));
    const allZero = ap.length > 0 && ap.every(([, v]) => v === 0);
    const operating = m.operating_point || m.operating_metrics;
    html = `<h3>Качество по кривой precision–recall</h3><div class="metrics">${ap.map(([k, v]) => `<div class="metric"><span>${esc(k.replace("metrics/", "").replace("(B)", ""))}</span><strong>${v.toFixed(3)}</strong></div>`).join("")}</div><p class="hint">mAP рассчитана при техническом cutoff ${m.ap_confidence_cutoff ?? 0.001} с изменением порога уверенности. Эти значения не являются precision / recall рабочего режима.</p>${allZero ? '<p class="callout warning">mAP = 0: правильных обнаружений на выбранной выборке не получено.</p>' : ""}<h3>Рабочий режим</h3><p class="hint">Раздел ${esc(m.split)} · confidence ${m.operating_confidence} · IoU совпадения ${m.matching_iou} · NMS IoU ${m.nms_iou ?? "—"}.</p>${
      operating
        ? `<div class="metrics">${[
            ["Precision", operating.precision],
            ["Recall", operating.recall],
            ["Macro F1", operating.macro_f1],
          ]
            .map(
              ([k, v]) =>
                `<div class="metric"><span>${k}</span><strong>${v.toFixed(3)}</strong></div>`,
            )
            .join(
              "",
            )}</div><p>TP: ${operating.tp} · FP: ${operating.fp} · FN: ${operating.fn}</p>`
        : '<p class="hint">Эта историческая версия не содержит итоговых показателей рабочего порога. Галерея показывает сохранённые примеры ошибок; библиотечные precision / recall не подставляются вместо рабочих показателей.</p>'
    }<table><thead><tr><th>Класс</th><th>AP50</th><th>AP50–95</th>${operating?.per_class ? "<th>F1</th><th>Объектов</th>" : ""}</tr></thead><tbody>${m.per_class.map((c) => `<tr><td>${esc(c.name)}</td><td>${c.ap50.toFixed(3)}</td><td>${c.ap50_95.toFixed(3)}</td>${operating?.per_class ? `<td>${operating.per_class[c.id]?.f1.toFixed(3) ?? "—"}</td><td>${operating.per_class[c.id]?.support ?? "—"}</td>` : ""}</tr>`).join("")}</tbody></table><h3>Сохранённые примеры ошибок</h3><label class="check-row"><input type="checkbox" id="show-ground-truth" checked>Показать правильную разметку</label><p class="hint">Пунктирные зелёные рамки — разметка. Рамки модели нанесены на изображение. Галерея может содержать только часть ошибок; итоговые числа берутся из полного отчёта.</p><div class="gallery" id="error-gallery"></div><div id="error-pages"></div>`;
    const dialog = showResult(title, html);
    function gallery() {
      const page = listPage(m.errors, `errors-${a.id}`, 12);
      const showTruth = dialog.querySelector("#show-ground-truth").checked;
      dialog.querySelector("#error-gallery").innerHTML =
        page.rows
          .map((e) => {
            const image = annotated(
              a,
              e.image,
              showTruth ? e.ground_truth : [],
              names,
              `Лишних рамок: ${e.false_positives} · пропущенных: ${e.false_negatives}`,
              true,
            );
            return image.replace(" · без фигур", "");
          })
          .join("") ||
        empty("Лишних и пропущенных рамок на выбранном пороге нет.");
      dialog.querySelector("#error-pages").innerHTML = page.navigation;
      wirePages(dialog.querySelector("#error-pages"), gallery);
    }
    dialog.querySelector("#show-ground-truth").onchange = gallery;
    gallery();
    return;
  }
  if (a.kind === "predict") {
    const records = await (
      await fetch(artifactUrl(a, "predictions.json"))
    ).json();
    const detections = records.reduce((n, r) => n + r.boxes.length, 0);
    const training = state.artifacts.find((x) => x.id === m.training_id);
    const dataset = state.artifacts.find(
      (x) => x.id === training?.manifest.metadata.dataset_id,
    );
    const names = m.class_names || dataset?.manifest.metadata.names || [];
    const pictures = files.filter((p) => /\.(png|jpe?g)$/i.test(p));
    const videos = files.filter(
      (p) => /\.(mp4|avi|mov|mkv)$/i.test(p) && !p.endsWith("-preview.mp4"),
    );
    html = `<p>${m.frames} кадров · найдено рамок: ${detections} · уверенность ${m.confidence}</p>${!detections ? '<div class="callout warning">Модель не нашла фигуры при выбранном пороге. Изображения ниже показаны без рамок. Проверьте качество обучения; снижение порога может добавить ложные обнаружения.</div>' : ""}<div class="gallery" id="prediction-gallery"></div><div id="prediction-pages"></div><a href="${artifactUrl(a, "predictions.json")}" target="_blank">Все рамки и уверенности</a>`;
    const dialog = showResult(title, html);
    function gallery() {
      const page = listPage([...pictures, ...videos], `prediction-${a.id}`, 12);
      dialog.querySelector("#prediction-gallery").innerHTML = page.rows
        .map((p) => {
          if (pictures.includes(p)) {
            const r = imagePrediction(records, p);
            const caption = `${p.split("/").at(-1)} · рамок: ${r?.boxes.length ?? "—"}`;
            const labels = (r?.classes || []).map((id) => esc(names[id] ?? id));
            return `<figure><img loading="lazy" src="${artifactUrl(a, p)}"><figcaption>${esc(caption)}${labels.length ? "<br>" + labels.join(", ") : ""}</figcaption></figure>`;
          }
          const preview = p.replace(/\.[^.]+$/, "-preview.mp4");
          const playable = files.includes(preview) ? preview : p;
          return `<figure><video controls style="width:100%" src="${artifactUrl(a, playable)}"></video><figcaption><a href="${artifactUrl(a, playable)}" download>Скачать видео</a> · <a href="${artifactUrl(a, p)}" download>Исходный результат</a></figcaption></figure>`;
        })
        .join("");
      dialog.querySelector("#prediction-pages").innerHTML = page.navigation;
      wirePages(dialog.querySelector("#prediction-pages"), gallery);
    }
    gallery();
    return;
  }
  const dialog = showResult(title, html);
  if (a.kind === "figure") {
    try {
      const viewer = await modelViewer(
        dialog.querySelector("#model-viewer"),
        artifactUrl(a, m.mesh),
      );
      dialog.querySelector("#neutral-model").onchange = (e) =>
        viewer.neutral(e.target.checked);
      dialog.addEventListener("close", () => viewer.dispose(), { once: true });
    } catch (e) {
      notice("Не удалось открыть 3D-просмотр: " + e.message, true);
    }
  }
}
