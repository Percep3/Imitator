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
  initNodeClicks();
  initTypeFilters();
  initSearch();
}

function showGraphError(message) {
  const box = document.getElementById('graph-error');
  box.textContent = message;
  box.hidden = false;
}

function edgeLabel(edge, otherId) {
  const otherNode = state.cy.getElementById(otherId);
  const metric = edge.data('metric') ? ` (${edge.data('metric')}=${edge.data('metric_value')})` : '';
  return `${edge.data('relation')} → ${otherNode.data('label')}${metric} [${edge.data('confidence')}]`;
}

function renderDetailPanel(node) {
  const panel = document.getElementById('detail-panel');
  const d = node.data();
  let html = `<h3>${d.label}</h3><p><strong>Tipo:</strong> ${d.node_type}</p>`;
  if (d.node_type === 'paper') {
    html += `<p><strong>Autores:</strong> ${(d.authors || []).join(', ') || '—'}</p>`;
    html += `<p><strong>Año:</strong> ${d.year ?? '—'}</p>`;
    html += `<p><strong>Venue:</strong> ${d.venue ?? '—'}</p>`;
    html += `<p><strong>DOI:</strong> ${d.doi ?? '—'}</p>`;
  }
  const connected = state.cy.edges(`[source = "${d.id}"], [target = "${d.id}"]`);
  html += `<p><strong>Relaciones (${connected.length}):</strong></p>`;
  connected.forEach(edge => {
    const otherId = edge.data('source') === d.id ? edge.data('target') : edge.data('source');
    html += `<div class="edge-row" data-node-id="${otherId}">${edgeLabel(edge, otherId)}</div>`;
  });
  panel.innerHTML = html;
  panel.querySelectorAll('.edge-row').forEach(row => {
    row.addEventListener('click', () => {
      const target = state.cy.getElementById(row.dataset.nodeId);
      state.cy.elements().removeClass('highlighted');
      target.addClass('highlighted');
      state.cy.animate({ center: { eles: target } }, { duration: 300 });
      renderDetailPanel(target);
    });
  });
}

function initNodeClicks() {
  state.cy.on('tap', 'node', evt => renderDetailPanel(evt.target));
}

function initTypeFilters() {
  document.querySelectorAll('#type-filters input[type=checkbox]').forEach(checkbox => {
    checkbox.addEventListener('change', () => {
      const type = checkbox.value;
      state.cy.nodes(`[node_type = "${type}"]`).style('display', checkbox.checked ? 'element' : 'none');
    });
  });
}

function initSearch() {
  document.getElementById('search-box').addEventListener('input', evt => {
    const term = evt.target.value.trim().toLowerCase();
    state.cy.elements().removeClass('highlighted');
    if (!term) return;
    const match = state.cy.nodes().filter(n => n.data('label').toLowerCase().includes(term))[0];
    if (match) {
      match.addClass('highlighted');
      state.cy.animate({ center: { eles: match } }, { duration: 300 });
    }
  });
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

function showTableError(message) {
  const box = document.getElementById('table-error');
  box.textContent = message;
  box.hidden = false;
}

function renderExtractionTable(rows) {
  const search = document.getElementById('table-search').value.trim().toLowerCase();
  const confidence = document.getElementById('confidence-filter').value;
  const filtered = rows.filter(row => {
    const matchesSearch = !search || [row.paper, row.dataset, row.baseline_method]
      .some(v => v && v.toLowerCase().includes(search));
    const matchesConfidence = !confidence || row.confidence === confidence;
    return matchesSearch && matchesConfidence;
  });
  const body = document.getElementById('extraction-table-body');
  body.innerHTML = filtered.map(row => `
    <tr>
      <td>${row.type}</td>
      <td>${row.paper}</td>
      <td>${row.dataset ?? row.baseline_method ?? '—'}</td>
      <td>${row.metric ?? '—'}</td>
      <td>${row.value ?? '—'}</td>
      <td>${row.confidence}</td>
    </tr>
  `).join('');
}

function initTableFilters() {
  document.getElementById('table-search').addEventListener('input', () => renderExtractionTable(state.extractionRows));
  document.getElementById('confidence-filter').addEventListener('change', () => renderExtractionTable(state.extractionRows));
}

function loadTable() {
  state.tableLoaded = true;
  Promise.all([
    fetch('../kb/extraction_table.json').then(res => {
      if (!res.ok) throw new Error(`HTTP ${res.status} extraction_table.json`);
      return res.json();
    }),
    fetch('../kb/research_gaps.md').then(res => {
      if (!res.ok) throw new Error(`HTTP ${res.status} research_gaps.md`);
      return res.text();
    }),
  ])
    .then(([rows, gapsMd]) => {
      state.extractionRows = rows;
      initTableFilters();
      renderExtractionTable(rows);
      document.getElementById('research-gaps').innerHTML = marked.parse(gapsMd);
    })
    .catch(err => {
      state.tableLoaded = false;
      showTableError(`No se pudo cargar la tabla: ${err.message}`);
    });
}

function init() {
  initTabs();
  loadGraph();
}

document.addEventListener('DOMContentLoaded', init);
