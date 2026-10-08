import { state, esc, api, artifactUrl, notice } from "./api.js";
import {
  showResult,
  empty,
  listPage,
  wirePages,
  trainingContext,
} from "./controls.js";
import { modelViewer } from "./viewer.js";
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
