import { isChosenCandidate } from "./candidate_choice.js";
import * as THREE from "three";
import { OrbitControls } from "./vendor/three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "./vendor/three/examples/jsm/loaders/GLTFLoader.js";
import { api } from "./api.js";

export function mountComparison(
  container,
  catalog,
  identifier,
  initialChoices,
  persistChoices,
  showArtifact,
) {
  let disposed = false,
    saving = false;
  const $ = (id) => container.querySelector(`[id="${id}"]`);
  const params = new URLSearchParams();
  const scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, 0.4));
  const key = new THREE.DirectionalLight(0xffffff, 2);
  key.position.set(-2, 3, 4);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xd6e4ff, 0.5);
  fill.position.set(3, 1, -2);
  scene.add(fill);
  const camera = new THREE.PerspectiveCamera(35, 1, 0.001, 100);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  $("modelArea").append(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = false;
  controls.enabled = false;
  const loader = new GLTFLoader();
  let figure = null,
    frames = [],
    frameIndex = 0,
    mode = "geometry",
    referenceMode = "mask",
    free = false,
    token = 0,
    loaded = new Map(),
    enabled = new Set();
  let favorites = Object.fromEntries(
    initialChoices.map((c) => [c.object + ":" + c.source_video_sha256, c]),
  );
  async function changeChoices(next) {
    if (saving || disposed) return;
    saving = true;
    container
      .querySelectorAll(
        ".choose, [data-separate-choice], #selectionList button",
      )
      .forEach((b) => (b.disabled = true));
    try {
      await persistChoices(Object.values(next));
      if (!disposed) {
        favorites = next;
        updateFavorites();
      }
    } catch (error) {
      if (!disposed)
        $("status").textContent = "Выбор не сохранён: " + error.message;
    } finally {
      saving = false;
      container
        .querySelectorAll(
          ".choose, [data-separate-choice], #selectionList button",
        )
        .forEach((b) => (b.disabled = false));
    }
  }
  for (const choice of Object.values(favorites)) {
    const entry = catalog.find((e) => e.object === choice.object);
    if (entry) choice.name = entry.name;
  }
  const pending = new Map();
  const favoriteKey = () => figure.object + ":" + figure.source_video_sha256;
  const chosen = () => favorites[favoriteKey()];
  function dispose(root) {
    const geometries = new Set(),
      materials = new Set(),
      textures = new Set();
    root.traverse((n) => {
      if (n.isMesh) {
        geometries.add(n.geometry);
        for (const m of (n.userData.materials || [n.material]).flat()) {
          if (!m) continue;
          materials.add(m);
          for (const v of Object.values(m)) if (v?.isTexture) textures.add(v);
        }
      }
    });
    for (const x of geometries) x.dispose();
    for (const x of materials) x.dispose();
    for (const x of textures) x.dispose();
    scene.remove(root);
  }
  function setMaterial(root) {
    root.traverse((n) => {
      if (n.isMesh) n.material = n.userData[mode];
    });
  }
  function projectionForFrame() {
    const frame = frames[frameIndex];
    camera.matrixAutoUpdate = false;
    camera.matrixWorldAutoUpdate = false;
    camera.projectionMatrix.set(...frame.projection.flat());
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    camera.matrixWorldInverse.set(...frame.view.flat());
    camera.matrixWorld.copy(camera.matrixWorldInverse).invert();
  }
  function draw() {
    if (!figure) return;
    const parent = $("modelArea").getBoundingClientRect();
    if (!parent.width || !parent.height) return;
    const w = Math.round(parent.width),
      h = Math.round(parent.height);
    renderer.setSize(w, h, false);
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, w, h);
    renderer.setClearColor(0x181d25, 0);
    renderer.clear();
    renderer.setScissorTest(true);
    const frame = frames[frameIndex];
    const aspect = frame.width / frame.height;
    if (!free) projectionForFrame();
    let rendered = 0;
    for (const entry of figure.candidates) {
      const root = loaded.get(entry.id);
      if (!root || !enabled.has(entry.id)) continue;
      const card = $("card-" + entry.id);
      if (!card) continue;
      const rect = card.querySelector(".model-window").getBoundingClientRect();
      const width = Math.min(rect.width, rect.height * aspect),
        height = width / aspect;
      const x = Math.round(rect.left - parent.left + (rect.width - width) / 2),
        y = Math.round(
          parent.bottom - rect.bottom + (rect.height - height) / 2,
        );
      for (const other of loaded.values()) other.visible = other === root;
      if (free) {
        camera.aspect = aspect;
        camera.updateProjectionMatrix();
      }
      renderer.setViewport(x, y, Math.round(width), Math.round(height));
      renderer.setScissor(x, y, Math.round(width), Math.round(height));
      renderer.setClearColor(
        mode === "silhouette"
          ? 0x000000
          : new THREE.Color(
              getComputedStyle(container).getPropertyValue("--bg").trim(),
            ),
        1,
      );
      renderer.clear();
      renderer.render(scene, camera);
      rendered++;
    }
    for (const root of loaded.values()) root.visible = true;
    renderer.setScissorTest(false);
    $("status").textContent =
      `Показано ${rendered} из ${enabled.size} выбранных кандидатов · ${free ? "свободное вращение" : "ракурс исходного кадра"}`;
  }
  function updateReference() {
    const f = frames[frameIndex];
    $("frame").value = String(frameIndex);
    $("frameSlider").value = frameIndex;
    $("referenceImage").src = f[referenceMode];
    $("referenceImage").alt =
      (referenceMode === "mask" ? "Маска" : "Фото") +
      ` исходного кадра ${f.source_frame} — ${figure.name}`;
    const time = f.seconds === null ? "" : ` · ${f.seconds.toFixed(2)} с`;
    $("frameInfo").textContent =
      `${frameIndex + 1}/${frames.length} · исходный кадр ${f.source_frame}${time} · ${f.split === "fit" ? "кадр обучения" : "validation, поза по орбите"}`;
    $("previous").disabled = frameIndex === 0;
    $("next").disabled = frameIndex === frames.length - 1;
    $("alignment").textContent = free
      ? "Свободный ракурс: он сейчас не совпадает с маской. Нажмите «Ракурс маски»."
      : "Поворот, положение и масштаб моделей задаёт сохранённая камера этого кадра. Независимое растяжение или подгонка моделей не применяются.";
    for (const image of $("modelGrid").querySelectorAll(".mask-contour")) {
      image.src = f.contour;
      image.hidden = free || !$("overlay").checked;
    }
    draw();
  }
  function setFrame(index) {
    frameIndex = Math.max(0, Math.min(frames.length - 1, index));
    free = false;
    controls.enabled = false;
    $("free").setAttribute("aria-pressed", "false");
    updateReference();
  }
  function updateFavorites() {
    updateSelection();
    if (!figure) return;
    const selected = [...figure.candidates, ...(figure.unavailable || [])].find(
      (e) => isChosenCandidate(e, chosen()),
    );
    $("favorite").textContent = selected
      ? `Мой выбор: ${selected.label}. Сохранён в проекте.`
      : "Можно отметить удачный вариант кнопкой «Мой выбор». Выбор сохраняется на сервере.";
    for (const entry of figure.candidates) {
      const card = $("card-" + entry.id);
      if (card) {
        const favorite = selected?.id === entry.id;
        card.classList.toggle("favorite", favorite);
        card.querySelector(".choose").textContent = favorite
          ? "✓ Мой выбор"
          : "Мой выбор";
      }
    }
  }
  function updateSelection() {
    const entries = Object.entries(favorites);
    $("selectionCount").textContent = entries.length;
    $("export").disabled = !entries.length;
    $("selectionList").replaceChildren(
      ...entries.map(([key, choice]) => {
        const item = document.createElement("li");
        const text = document.createElement("span");
        text.textContent = `${choice.name} — ${choice.label} `;
        const remove = document.createElement("button");
        remove.textContent = "Убрать из набора";
        remove.onclick = () => {
          const next = { ...favorites };
          delete next[key];
          changeChoices(next);
        };
        item.append(text, remove);
        return item;
      }),
    );
    $("selectionHint").textContent = entries.length
      ? "Новые сцены используют только перечисленные фигурки и точные версии моделей. Контрольные суммы проверяются при выборе и запуске."
      : "Отметьте «Мой выбор» у нужных моделей. Неотмеченные фигурки в набор не войдут.";
  }
  function createCards() {
    $("modelGrid").replaceChildren();
    for (const entry of figure.candidates.filter((e) => enabled.has(e.id))) {
      const card = document.createElement("article");
      card.className = "model-card" + (loaded.has(entry.id) ? "" : " loading");
      card.id = "card-" + entry.id;
      const header = document.createElement("div");
      header.className = "card-header";
      const title = document.createElement("h3");
      title.textContent = entry.label;
      header.append(title);
      const metric = document.createElement("p");
      metric.textContent = entry.metrics
        ? `IoU ${entry.metrics.iou.toFixed(3)} · полнота ${entry.metrics.foreground_recall.toFixed(3)}`
        : entry.review_status === "diagnostic"
          ? "Вариант с известными дефектами"
          : entry.review_status === "repair_candidate"
            ? "Ремонт / контраст · оценка маской не пересчитана"
            : "Ранее сохранённый кандидат";
      header.append(metric);
      const window = document.createElement("div");
      window.className = "model-window";
      const contour = new Image();
      contour.className = "mask-contour";
      contour.alt = "Контур исходной маски поверх 3D";
      window.append(contour);
      const footer = document.createElement("div");
      footer.className = "card-footer";
      const note = document.createElement("p");
      note.textContent = entry.note;
      footer.append(note);
      const links = document.createElement("div");
      links.className = "card-links";
      const download = document.createElement("a");
      download.href = entry.mesh;
      download.download = entry.download_name;
      download.textContent = "Скачать GLB";
      const choose = document.createElement("button");
      choose.className = "choose";
      choose.textContent = "Мой выбор";
      choose.onclick = () => {
        const next = { ...favorites };
        if (isChosenCandidate(entry, chosen())) delete next[favoriteKey()];
        else {
          for (const [key, choice] of Object.entries(next))
            if (choice.object === figure.object) delete next[key];
          next[favoriteKey()] = {
            object: figure.object,
            name: figure.name,
            id: entry.id,
            label: entry.label,
            mesh_sha256: entry.mesh_sha256,
            source_video_sha256: figure.source_video_sha256,
            chosen_at: new Date().toISOString(),
          };
        }
        changeChoices(next);
      };
      links.append(download, choose);
      footer.append(links);
      card.append(header, window, footer);
      $("modelGrid").append(card);
    }
    updateFavorites();
    updateReference();
  }
  function load(entry, currentToken) {
    const key = currentToken + ":" + entry.id;
    if (pending.has(key)) return pending.get(key);
    const task = loadMesh(entry, currentToken).finally(() =>
      pending.delete(key),
    );
    pending.set(key, task);
    return task;
  }
  async function loadMesh(entry, currentToken) {
    if (loaded.has(entry.id)) return;
    const gltf = await loader.loadAsync(
        entry.mesh + "?sha256=" + entry.mesh_sha256,
      ),
      root = gltf.scene;
    if (disposed || currentToken !== token || !enabled.has(entry.id)) {
      dispose(root);
      return;
    }
    root.applyMatrix4(
      new THREE.Matrix4().set(...entry.world_from_model.flat()),
    );
    root.traverse((n) => {
      if (n.isMesh) {
        if (!n.geometry.getAttribute("normal"))
          n.geometry.computeVertexNormals();
        const original = Array.isArray(n.material) ? n.material : [n.material];
        const grey = new THREE.MeshStandardMaterial({
          color: 0xaaaaaa,
          roughness: 0.8,
          metalness: 0,
          side: THREE.DoubleSide,
        });
        const white = new THREE.MeshBasicMaterial({
          color: 0xffffff,
          side: THREE.DoubleSide,
        });
        const material = original.map((m) => {
          if (!m.map && !m.vertexColors) {
            const plastic = m.clone();
            plastic.side = THREE.DoubleSide;
            return plastic;
          }
          return new THREE.MeshBasicMaterial({
            map: m.map,
            color: m.color,
            vertexColors: m.vertexColors,
            side: THREE.DoubleSide,
          });
        });
        n.userData.geometry = grey;
        n.userData.silhouette = white;
        n.userData.material = material.length === 1 ? material[0] : material;
        n.userData.materials = [...original, grey, white, ...material];
      }
    });
    loaded.set(entry.id, root);
    scene.add(root);
    setMaterial(root);
    $("card-" + entry.id)?.classList.remove("loading");
    draw();
  }
  async function synchronize() {
    for (const [id, root] of loaded) {
      if (!enabled.has(id)) {
        dispose(root);
        loaded.delete(id);
      }
    }
    createCards();
    const currentToken = token;
    const results = await Promise.allSettled(
      figure.candidates
        .filter((e) => enabled.has(e.id))
        .map((e) => load(e, currentToken)),
    );
    if (currentToken !== token) return;
    const errors = results.filter((x) => x.status === "rejected");
    if (errors.length)
      $("status").textContent =
        "Не удалось загрузить сетку: " +
        errors.map((x) => x.reason.message).join(";");
  }
  async function selectFigure(object) {
    const currentToken = ++token;
    for (const root of loaded.values()) dispose(root);
    loaded.clear();
    $("modelGrid").replaceChildren();
    $("candidates").replaceChildren();
    $("separate-candidates").replaceChildren();
    $("referenceImage").removeAttribute("src");
    renderer.renderLists.dispose();
    const busyControls = [
      "frame",
      "frameSlider",
      "previous",
      "next",
      "free",
      "restore",
    ];
    for (const id of busyControls) $(id).disabled = true;
    const row = catalog.find((e) => e.object === object);
    $("status").textContent = "Загружаются кандидаты и маски…";
    let data;
    try {
      data = await api(
        `/comparison/figure/${encodeURIComponent(row.object)}?package=${encodeURIComponent(identifier)}`,
      );
    } catch (error) {
      if (currentToken === token)
        $("status").textContent =
          "Не удалось загрузить фигуру: " + error.message;
      return;
    }
    if (currentToken !== token) return;
    for (const id of busyControls) $(id).disabled = false;
    figure = data;
    frames = data.frames;
    frameIndex = Math.max(
      0,
      frames.findIndex((f) => String(f.index) === params.get("frame")),
    );
    free = false;
    controls.enabled = false;
    $("free").setAttribute("aria-pressed", "false");
    enabled = new Set(
      data.candidates.filter((e) => e.default).map((e) => e.id),
    );
    if (!enabled.size && data.candidates.length)
      enabled.add(data.candidates[0].id);
    renderSeparateCandidates();
    if (!frames.length) {
      for (const id of busyControls) $(id).disabled = true;
      $("frameSlider").max = 0;
      $("frameSlider").value = 0;
      $("modelGrid").replaceChildren();
      $("candidates").replaceChildren();
      $("referenceImage").removeAttribute("src");
      $("frame").replaceChildren();
      $("inventory").textContent =
        `Версий без общих камер: ${data.unavailable.length}`;
      $("availability").textContent =
        "Синхронное сравнение требует сохранённых камер. Эти версии можно открыть отдельно и выбрать ниже.";
      $("status").textContent = data.unavailable.length
        ? "Доступен отдельный просмотр моделей"
        : "Подготовьте или подключите кандидатов";
      updateSelection();
      return;
    }
    $("frame").replaceChildren(
      ...frames.map(
        (f, i) =>
          new Option(`Поза ${i + 1} · кадр ${f.source_frame}`, String(i)),
      ),
    );
    $("frameSlider").max = frames.length - 1;
    const diagnosticCount = data.candidates.filter(
      (e) => e.review_status === "diagnostic",
    ).length;
    $("inventory").textContent =
      `Кандидатов: ${data.candidates.length - diagnosticCount} · диагностических: ${diagnosticCount} · масок: ${frames.length}`;
    const mv = data.candidates.some((e) => e.method === "hunyuan_mv");
    $("availability").textContent =
      (mv
        ? "MV: есть сохранённый кандидат. "
        : "MV: в этом наборе нет сохранённого кандидата с привязкой к этим камерам. ") +
      (diagnosticCount
        ? "Некоторые native / depth fusion ранее исключены из-за дефектов. Их можно открыть переключателем ниже. "
        : "") +
      "Ремонт основания и усиление текстуры — отдельные варианты; они не исправляют неполный корпус.";
    renderCandidateChoices();
    await synchronize();
  }
  function renderSeparateCandidates() {
    const root = $("separate-candidates");
    root.replaceChildren();
    for (const entry of figure.unavailable || []) {
      const row = document.createElement("div");
      row.className = "actions source-row";
      const text = document.createElement("span");
      text.textContent = `${entry.label} · ${entry.reason}`;
      const view = document.createElement("button");
      view.textContent = "Открыть отдельно";
      view.onclick = () => showArtifact(entry.id);
      const choose = document.createElement("button");
      choose.dataset.separateChoice = entry.id;
      choose.textContent = isChosenCandidate(entry, chosen())
        ? "✓ Мой выбор"
        : "Мой выбор";
      choose.onclick = async () => {
        const next = Object.fromEntries(
          Object.entries(favorites).filter(
            ([, c]) => c.object !== figure.object,
          ),
        );
        if (!isChosenCandidate(entry, chosen()))
          next[favoriteKey()] = {
            object: figure.object,
            name: figure.name,
            id: entry.id,
            label: entry.label,
            mesh_sha256: entry.mesh_sha256,
            source_video_sha256: figure.source_video_sha256,
          };
        await changeChoices(next);
        if (!disposed) renderSeparateCandidates();
      };
      row.append(text, view, choose);
      root.append(row);
    }
  }
  function renderCandidateChoices() {
    $("candidates").replaceChildren();
    for (const entry of figure.candidates.filter(
      (e) => $("showDiagnostics").checked || e.review_status !== "diagnostic",
    )) {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = enabled.has(entry.id);
      input.dataset.candidate = entry.id;
      input.onchange = () => {
        if (!input.checked && enabled.size === 1) {
          input.checked = true;
          return;
        }
        if (input.checked) enabled.add(entry.id);
        else enabled.delete(entry.id);
        synchronize();
      };
      label.append(
        input,
        document.createTextNode(
          entry.label +
            (entry.review_status === "diagnostic" ? " · есть дефекты" : ""),
        ),
      );
      label.title = entry.note;
      $("candidates").append(label);
    }
  }
  $("showDiagnostics").onchange = () => {
    if (!figure || !frames.length) return;
    if (!$("showDiagnostics").checked) {
      for (const entry of figure.candidates)
        if (entry.review_status === "diagnostic") enabled.delete(entry.id);
    }
    if (!enabled.size)
      enabled.add(
        figure.candidates.find((e) => e.review_status !== "diagnostic")?.id ||
          figure.candidates[0]?.id,
      );
    renderCandidateChoices();
    synchronize();
  };
  $("object").replaceChildren(
    ...catalog.map((e) => new Option(e.name, e.object)),
  );
  $("object").value = catalog.some((e) => e.object === params.get("object"))
    ? params.get("object")
    : catalog[0].object;
  $("object").onchange = () => selectFigure($("object").value);
  $("frame").onchange = () => setFrame(Number($("frame").value));
  $("frameSlider").oninput = () => setFrame(Number($("frameSlider").value));
  $("previous").onclick = () => setFrame(frameIndex - 1);
  $("next").onclick = () => setFrame(frameIndex + 1);
  $("overlay").onchange = updateReference;
  for (const name of ["geometry", "material", "silhouette"])
    $(name).onclick = () => {
      mode = name;
      for (const n of ["geometry", "material", "silhouette"])
        $(n).setAttribute("aria-pressed", String(n === mode));
      for (const root of loaded.values()) setMaterial(root);
      draw();
    };
  for (const [button, name] of [
    ["showMask", "mask"],
    ["showPhoto", "photo"],
  ])
    $(button).onclick = () => {
      referenceMode = name;
      $("showMask").setAttribute("aria-pressed", String(name === "mask"));
      $("showPhoto").setAttribute("aria-pressed", String(name === "photo"));
      updateReference();
    };
  $("free").onclick = () => {
    free = !free;
    controls.enabled = free;
    $("free").setAttribute("aria-pressed", String(free));
    if (free) {
      camera.matrixAutoUpdate = true;
      camera.matrixWorldAutoUpdate = true;
      const bounds = figure.reference_bounds;
      const center = new THREE.Vector3(...bounds[0])
        .add(new THREE.Vector3(...bounds[1]))
        .multiplyScalar(0.5);
      const size = new THREE.Vector3(...bounds[1]).sub(
        new THREE.Vector3(...bounds[0]),
      );
      camera.position
        .copy(center)
        .add(
          new THREE.Vector3(0, 0.2, 2.5).multiplyScalar(
            Math.max(size.x, size.y, size.z),
          ),
        );
      controls.target.copy(center);
      controls.update();
    }
    updateReference();
  };
  $("restore").onclick = () => setFrame(frameIndex);
  $("export").onclick = () => {
    const blob = new Blob(
      [
        JSON.stringify(
          { schema_version: 1, choices: Object.values(favorites) },
          null,
          2,
        ),
      ],
      { type: "application/json" },
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "figure-comparison-choices.json";
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  };
  controls.addEventListener("change", () => {
    if (free && !disposed) draw();
  });
  const observer = new ResizeObserver(() => {
    if (!disposed) draw();
  });
  observer.observe($("modelArea"));
  updateSelection();
  selectFigure($("object").value).catch((error) => {
    if (!disposed) $("status").textContent = error.message;
  });
  return () => {
    disposed = true;
    ++token;
    observer.disconnect();
    controls.dispose();
    for (const root of loaded.values()) dispose(root);
    loaded.clear();
    $("modelGrid").replaceChildren();
    $("candidates").replaceChildren();
    $("separate-candidates").replaceChildren();
    $("referenceImage").removeAttribute("src");
    renderer.renderLists.dispose();
    renderer.dispose();
    renderer.forceContextLoss();
    renderer.domElement.remove();
  };
}
