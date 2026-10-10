import { api, artifactUrl, esc, notice } from "./api.js";
import { showResult } from "./controls.js";
import { sceneViewer } from "./scene_viewer.js";

export async function showLocalization(artifact) {
  const data = await (await fetch(artifactUrl(artifact, "scene.json"))).json();
  const m = artifact.manifest.metadata;
  const dialog = showResult(
    "Координаты и 3D-сцена",
    `<p>Обработано кадров: ${m.processed} / ${m.frames} · оценок положения: ${m.estimated} · неоднозначных: ${m.uncertain}</p>
    <p class="hint">Зелёный круг — оценка основания, оранжевый — неоднозначная оценка. Поворот GLB является гипотезой подгонки по рамке. При перекрытии фигуры точность не гарантируется.</p>
    <div class="actions"><a href="${artifactUrl(artifact, "frames.jsonl")}" download>Координаты всех кадров (JSONL)</a><a href="${artifactUrl(artifact, "scene.json")}" download>Описание сцены (JSON)</a></div>
    <label class="field"><span class="label">Кадр</span><select id="scene-frame">${data.frames.map((f, i) => `<option value="${i}">${esc(f.source.split(/[\\/]/).at(-1))} · кадр ${f.frame_index}${f.timestamp_s == null ? "" : " · " + f.timestamp_s.toFixed(2) + " с"} · ${f.status === "processed" ? "поле найдено" : "поле не найдено"}</option>`).join("")}</select></label>
    <div id="scene-body"></div>`,
  );
  let viewer = null,
    generation = 0,
    closed = false;
  dialog.addEventListener(
    "close",
    () => {
      closed = true;
      generation++;
      viewer?.dispose();
    },
    { once: true },
  );
  async function show(index) {
    const current = ++generation;
    viewer?.dispose();
    viewer = null;
    const frame = data.frames[index];
    const root = dialog.querySelector("#scene-body");
    root.innerHTML = `${frame.error ? `<p class="callout warning">${esc(frame.error)}</p>` : ""}<div class="gallery"><figure><img alt="Проекция моделей на кадр" src="${artifactUrl(artifact, frame.overlay)}"><figcaption>Проекция моделей и оснований на исходный кадр</figcaption></figure>${frame.map ? `<figure><img alt="Положения фигур на карте" src="${artifactUrl(artifact, frame.map)}"><figcaption>Положения оснований на карте</figcaption></figure>` : ""}</div>
      ${frame.status === "processed" ? `<div class="viewer" id="scene-viewer"></div><label class="check-row"><input type="checkbox" id="scene-top">Вид сверху</label><p class="hint">X вправо, Y к верхнему краю исходной карты. Начало координат в центре поля. Размеры в метрах.</p><table><thead><tr><th>Фигура</th><th>X, мм</th><th>Y, мм</th><th>Радиус, мм</th><th>Разброс гипотез, мм</th><th>Статус</th></tr></thead><tbody>${frame.figures.map((f) => `<tr><td>${esc(data.models[f.class_id]?.name || f.class_id)}</td><td>${f.xy_m ? (f.xy_m[0] * 1000).toFixed(1) : "—"}</td><td>${f.xy_m ? (f.xy_m[1] * 1000).toFixed(1) : "—"}</td><td>${f.radius_m ? (f.radius_m * 1000).toFixed(1) : "—"}</td><td>${f.position_spread_m == null ? "—" : (f.position_spread_m * 1000).toFixed(1)}</td><td>${f.status === "estimated" ? "Оценено" : f.status === "uncertain" ? "Неоднозначно" : "Не определено"}</td></tr>`).join("")}</tbody></table>` : ""}`;
    if (frame.status !== "processed") return;
    try {
      const created = await sceneViewer(
        root.querySelector("#scene-viewer"),
        { ...data, board: frame.board || data.board },
        frame,
        (p) => artifactUrl(artifact, p),
      );
      if (closed || current !== generation) {
        created.dispose();
        return;
      }
      viewer = created;
      root.querySelector("#scene-top").onchange = (event) =>
        viewer.top(event.target.checked);
    } catch (error) {
      if (!closed && current === generation)
        notice("Не удалось открыть сцену: " + error.message, true);
    }
  }
  dialog.querySelector("#scene-frame").onchange = (event) =>
    show(Number(event.target.value));
  await show(0);
}
