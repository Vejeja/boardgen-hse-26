import { state, api, applyProject, refresh, save, notice, setPath } from "./api.js";
import { wireHelp } from "./controls.js";
import * as project from "./project.js";
function render() {
  document.querySelector("#app").innerHTML = project.render();
  project.wire(render, render, async (kind) => {
    await save();
    await api("/jobs", {method:"POST",body:JSON.stringify({kind,parameters:{}})});
  }, save);
  document.querySelectorAll("[data-bind]").forEach(control => {
    control.onchange = async () => {
      setPath(state.project, control.dataset.bind, control.value);
      try { await save(); } catch (error) { notice(error.message, true); }
    };
  });
  wireHelp();
}
applyProject(await api("/project"));
await refresh();
render();
