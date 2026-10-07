const response = await fetch("/api/project");
const state = await response.json();
document.querySelector("#project").textContent = JSON.stringify(state.project, null, 2);
