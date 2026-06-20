const state = {
  cy: null,
  tableLoaded: false,
  extractionRows: [],
};

function showTab(name) {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === name);
  });
  document.querySelectorAll('.tab-panel').forEach(panel => {
    panel.classList.toggle('active', panel.id === `tab-${name}`);
  });
  if (name === 'table' && !state.tableLoaded) {
    loadTable();
  }
}

function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => showTab(btn.dataset.tab));
  });
}

function loadGraph() {
  // implemented in Task 2
}

function loadTable() {
  // implemented in Task 5
}

function init() {
  initTabs();
  loadGraph();
}

document.addEventListener('DOMContentLoaded', init);
