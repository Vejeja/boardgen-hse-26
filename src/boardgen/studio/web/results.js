import { state, esc, api, artifactUrl, notice } from "./api.js";
import {
  showResult,
  empty,
  listPage,
  wirePages,
} from "./controls.js";
import { modelViewer } from "./viewer.js";
export async function showArtifact(id) {
  if (!state.artifacts.some((x) => x.id === id)) return;
  const a = await api(`/artifacts/${id}`);
  const m = a.manifest.metadata;
  const files = Object.keys(a.manifest.files);
  let html = "";
  let title = {
    figure: "3D-модель",
    frames: "Кадры и маски",
  }[a.kind];
  if (a.kind === "figure")
    html = `<p>${esc(m.name)} · ${m.faces.toLocaleString("ru")} граней · ${m.origin === "research-import" ? "прежние вычисления" : "результат обработки"}</p><div class="viewer" id="model-viewer"></div><label class="check-row"><input type="checkbox" id="neutral-model">Показать геометрию без текстуры</label><div class="actions"><a href="${artifactUrl(a, m.mesh)}" download>Скачать GLB</a><a href="${artifactUrl(a, "manifest.json")}" target="_blank">Происхождение и параметры</a></div>${m.research_note ? `<p class="hint">${esc(m.research_note)}</p>` : ""}`;
  if (a.kind === "frames")
    html = `<img style="max-width:100%" src="${artifactUrl(a, m.contact)}"><p class="hint">${m.fit.length} кадров для построения · ${m.heldout.length} для проверки</p><table><thead><tr><th>Кадр</th><th>Резкость</th><th>Граница кадра</th><th>Согласие масок (IoU)</th></tr></thead><tbody>${m.diagnostics.map((d) => `<tr><td>${d.view}</td><td>${d.sharpness.toFixed(1)}</td><td>${d.touches_border ? "Касается" : "Свободна"}</td><td>${d.cross_check_iou?.toFixed(3) ?? "—"}</td></tr>`).join("")}</tbody></table>`;
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
