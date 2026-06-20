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

function buildElements(graph) {
  const nodes = graph.nodes.map(n => ({ data: { ...n } }));
  const edges = graph.edges.map((e, i) => ({
    data: { ...e, id: `e${i}_${e.source}_${e.target}` },
  }));
  return [...nodes, ...edges];
}

const NODE_STYLE = {
  paper: { shape: 'rectangle', color: '#2563eb' },
  method: { shape: 'ellipse', color: '#16a34a' },
  dataset: { shape: 'diamond', color: '#ea580c' },
  gap: { shape: 'triangle', color: '#dc2626' },
};

const EDGE_LINE_STYLE = {
  EXTRACTED: 'solid',
  INFERRED: 'dashed',
  AMBIGUOUS: 'dotted',
};

function cytoscapeStyle() {
  const style = [
    {
      selector: 'node',
      style: {
        label: 'data(label)',
        'font-size': 9,
        'text-wrap': 'wrap',
        'text-max-width': 80,
        width: 28,
        height: 28,
        color: '#1f2937',
      },
    },
    {
      selector: 'edge',
      style: {
        width: 1.5,
        'line-color': '#9ca3af',
        'target-arrow-color': '#9ca3af',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
      },
    },
    { selector: '.highlighted', style: { 'border-width': 4, 'border-color': '#facc15' } },
  ];
  for (const [type, { shape, color }] of Object.entries(NODE_STYLE)) {
    style.push({
      selector: `node[node_type = "${type}"]`,
      style: { shape, 'background-color': color },
    });
  }
  for (const [confidence, lineStyle] of Object.entries(EDGE_LINE_STYLE)) {
    style.push({
      selector: `edge[confidence = "${confidence}"]`,
      style: { 'line-style': lineStyle },
    });
  }
  return style;
}

function initCytoscape(elements) {
  state.cy = cytoscape({
    container: document.getElementById('cy'),
    elements,
    style: cytoscapeStyle(),
    layout: { name: 'cose', animate: false },
  });
}

function showGraphError(message) {
  const box = document.getElementById('graph-error');
  box.textContent = message;
  box.hidden = false;
}

function loadGraph() {
  fetch('../graph.json')
    .then(res => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    })
    .then(graph => {
      initCytoscape(buildElements(graph));
    })
    .catch(err => showGraphError(`No se pudo cargar graph.json: ${err.message}`));
}

function loadTable() {
  // implemented in Task 5
}

function init() {
  initTabs();
  loadGraph();
}

document.addEventListener('DOMContentLoaded', init);
