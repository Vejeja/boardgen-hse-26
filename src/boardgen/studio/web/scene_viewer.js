export async function sceneViewer(container, data, frame, fileUrl) {
  const THREE = await import("three");
  const [{ OrbitControls }, { GLTFLoader }] = await Promise.all([
    import("./vendor/three/examples/jsm/controls/OrbitControls.js"),
    import("./vendor/three/examples/jsm/loaders/GLTFLoader.js"),
  ]);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#e8eeeb");
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.append(renderer.domElement);
  const span = Math.max(data.board.width_m, data.board.height_m);
  const perspective = new THREE.PerspectiveCamera(40, 1, 0.001, span * 100);
  perspective.position.set(span * 0.9, span * 1.2, span * 1.3);
  perspective.lookAt(0, 0, 0);
  const top = new THREE.OrthographicCamera(
    -span,
    span,
    span,
    -span,
    0.001,
    span * 100,
  );
  top.position.set(0, span * 3, 0);
  top.up.set(0, 0, -1);
  top.lookAt(0, 0, 0);
  let camera = perspective;
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0);
  controls.update();
  const resources = new Set();
  let disposed = false;
  const track = (object) => {
    object.traverse((node) => {
      if (!node.isMesh) return;
      resources.add(node.geometry);
      for (const material of [node.material].flat()) {
        resources.add(material);
        for (const value of Object.values(material))
          if (value?.isTexture) resources.add(value);
      }
    });
  };
  function draw() {
    if (disposed) return;
    const w = container.clientWidth,
      h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    perspective.aspect = w / h;
    perspective.updateProjectionMatrix();
    top.left = (-span * 0.65 * w) / h;
    top.right = (span * 0.65 * w) / h;
    top.top = span * 0.65;
    top.bottom = -span * 0.65;
    top.updateProjectionMatrix();
    renderer.render(scene, camera);
  }
  const observer = new ResizeObserver(draw);
  observer.observe(container);
  controls.addEventListener("change", draw);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8c978d, 2));
  const light = new THREE.DirectionalLight(0xffffff, 2);
  light.position.set(-span, span * 3, span);
  scene.add(light);
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    observer.disconnect();
    controls.dispose();
    for (const resource of resources) resource.dispose();
    renderer.dispose();
    container.replaceChildren();
  };
  try {
    const texture = await new THREE.TextureLoader().loadAsync(
      fileUrl(data.board.file),
    );
    resources.add(texture);
    texture.colorSpace = THREE.SRGBColorSpace;
    const board = new THREE.Mesh(
      new THREE.PlaneGeometry(data.board.width_m, data.board.height_m),
      new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide }),
    );
    board.rotation.x = -Math.PI / 2;
    scene.add(board);
    track(board);
    const loader = new GLTFLoader();
    const models = new Map();
    for (const item of frame.figures) {
      if (!item.placement_xy_m) continue;
      const meta = data.models[item.class_id];
      if (!models.has(item.class_id)) {
        const gltf = await loader.loadAsync(fileUrl(meta.file));
        track(gltf.scene);
        models.set(item.class_id, gltf.scene);
      }
      const root = models.get(item.class_id).clone(true);
      const box = new THREE.Box3().setFromObject(root);
      const extent = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      root.position.sub(new THREE.Vector3(center.x, box.min.y, center.z));
      const placement = new THREE.Group();
      placement.add(root);
      placement.scale.setScalar(meta.height_m / extent.y);
      placement.position.set(
        item.placement_xy_m[0],
        0.00005,
        -item.placement_xy_m[1],
      );
      placement.rotation.y = (item.yaw_deg * Math.PI) / 180;
      scene.add(placement);
      const circle = new THREE.Mesh(
        new THREE.RingGeometry(item.radius_m * 0.96, item.radius_m, 64),
        new THREE.MeshBasicMaterial({
          color: item.status === "estimated" ? 0x27934a : 0xe49421,
          side: THREE.DoubleSide,
        }),
      );
      circle.rotation.x = -Math.PI / 2;
      circle.position.set(item.xy_m[0], 0.0002, -item.xy_m[1]);
      scene.add(circle);
      track(circle);
    }
    draw();
  } catch (error) {
    dispose();
    throw error;
  }
  return {
    top(value) {
      if (disposed) return;
      camera = value ? top : perspective;
      controls.object = camera;
      controls.enableRotate = !value;
      controls.update();
      draw();
    },
    dispose,
  };
}
