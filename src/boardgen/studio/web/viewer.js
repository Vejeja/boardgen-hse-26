export async function modelViewer(container, url) {
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
  const camera = new THREE.PerspectiveCamera(35, 1, 0.001, 100);
  camera.position.set(1.3, 0.9, 2);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.5, 0);
  controls.addEventListener("change", draw);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8f9a91, 2));
  const light = new THREE.DirectionalLight(0xffffff, 2);
  light.position.set(-2, 4, 3);
  scene.add(light);
  let root,
    disposed = false;
  const originals = new Map();
  const previews = new Map();
  const neutral = new THREE.MeshStandardMaterial({
    color: 0xa4aaa5,
    roughness: 0.75,
  });
  function draw() {
    if (disposed) return;
    const w = container.clientWidth,
      h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.render(scene, camera);
  }
  const observer = new ResizeObserver(draw);
  observer.observe(container);
  await new Promise((resolve, reject) =>
    new GLTFLoader().load(
      url,
      (gltf) => {
        if (disposed) return resolve();
        root = gltf.scene;
        const box = new THREE.Box3().setFromObject(root);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        root.position.sub(center);
        root.scale.setScalar(1 / Math.max(size.x, size.y, size.z));
        controls.target.set(0, 0, 0);
        camera.position.set(1.3, 0.75, 1.8);
        root.traverse((n) => {
          if (n.isMesh) {
            originals.set(n, n.material);
            const convert = (mat) =>
              new THREE.MeshBasicMaterial({
                color: mat.color ?? 0xffffff,
                map: mat.map,
                vertexColors: mat.vertexColors,
                side: mat.side,
                transparent: mat.transparent,
                opacity: mat.opacity,
                alphaTest: mat.alphaTest,
              });
            const material = Array.isArray(n.material)
              ? n.material.map(convert)
              : convert(n.material);
            previews.set(n, material);
            n.material = material;
          }
        });
        scene.add(root);
        draw();
        resolve();
      },
      undefined,
      reject,
    ),
  );
  return {
    neutral(value) {
      root?.traverse((n) => {
        if (n.isMesh) n.material = value ? neutral : previews.get(n);
      });
      draw();
    },
    dispose() {
      disposed = true;
      observer.disconnect();
      controls.dispose();
      root?.traverse((n) => {
        if (n.isMesh) {
          n.geometry.dispose();
          for (const mat of [originals.get(n)].flat()) {
            for (const value of Object.values(mat || {}))
              if (value?.isTexture) value.dispose();
            mat?.dispose();
          }
          for (const mat of [previews.get(n)].flat()) mat?.dispose();
        }
      });
      neutral.dispose();
      renderer.dispose();
      container.replaceChildren();
    },
  };
}
