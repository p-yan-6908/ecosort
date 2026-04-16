// ========== State ==========
let selectedFile = null;
let currentTab = 'photo';
let predictionHistory = [];
let batchFiles = []; // Array of { file, id, preview }
let isBatchProcessing = false;

// ========== Constants ==========
const HISTORY_KEY = 'ecosort_history';
const MAX_HISTORY = 10;
const MAX_BATCH_FILES = 10;

// ========== DOM ==========
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const dropZone = $('#drop-zone');
const fileInput = $('#file-input');
const cameraInput = $('#camera-input');
const btnBrowse = $('#btn-browse');
const btnCamera = $('#btn-camera');
const previewContainer = $('#preview-container');
const previewImage = $('#preview-image');
const clearBtn = $('#clear-btn');
const classifyBtn = $('#classify-btn');
const btnContent = $('#btn-content');
const btnLoader = $('#btn-loader');
const resultContainer = $('#result-container');
const errorContainer = $('#error-container');
const errorMessage = $('#error-message');

// Batch DOM elements
const batchDropZone = $('#batch-drop-zone');
const batchFileInput = $('#batch-file-input');
const btnBatchBrowse = $('#btn-batch-browse');
const batchFileList = $('#batch-file-list');
const batchFilesContainer = $('#batch-files');
const batchCount = $('#batch-count');
const btnClearBatch = $('#btn-clear-batch');
const btnClassifyBatch = $('#btn-classify-batch');
const btnBatchContent = $('#btn-batch-content');
const btnBatchLoader = $('#btn-batch-loader');
const batchProgress = $('#batch-progress');
const batchProgressFill = $('#batch-progress-fill');
const batchProgressLabel = $('#batch-progress-text');
const batchProgressText = $('#batch-progress-label');
const batchResults = $('#batch-results');
const batchResultsList = $('#batch-results-list');
const btnClearBatchResults = $('#btn-clear-batch-results');

// ========== Category colors (earthy tones) ==========
const CATEGORY_COLORS = {
  blue_bin: '#5a7fa8',
  green_bin: '#6b9b6b',
  garbage: '#6a6a6a',
  hazardous: '#b87070',
  e_waste: '#8a7fa8',
  yard_waste: '#9ab870',
};

const CATEGORY_ICONS = {
  blue_bin: '♻️',
  green_bin: '🌿',
  garbage: '🗑️',
  hazardous: '⚠️',
  e_waste: '💻',
  yard_waste: '🍂',
};

// ========== Sorting tips ==========
const SORTING_TIPS = {
  blue_bin: [
    'Rinse containers before recycling',
    'Flatten cardboard boxes',
    'Remove caps from bottles',
    'No black plastic or Styrofoam in most municipalities',
  ],
  green_bin: [
    'Use certified compostable bags',
    'No plastic bags in green bin',
    'Include meat, bones, dairy',
    'Soiled paper towels and napkins go here',
  ],
  garbage: [
    'Last resort — check other bins first',
    'Compostable plastics go to garbage, not green bin',
    'Broken ceramics and mirrors go here',
    'Bag all garbage securely',
  ],
  hazardous: [
    'Take to designated drop-off depots',
    'Never put in regular garbage or recycling',
    'Keep in original containers',
    'Check municipality for depot hours',
  ],
  e_waste: [
    'Drop off at e-waste depots or retail stores',
    'Wipe personal data before disposal',
    'Cables and chargers count as e-waste',
    'Many retailers accept old electronics',
  ],
  yard_waste: [
    'Use paper yard waste bags or open containers',
    'No plastic bags',
    'Branches must be under 10cm diameter',
    'Check seasonal curbside collection dates',
  ],
};

// ========== History Management ==========
function loadHistory() {
  try {
    const saved = localStorage.getItem(HISTORY_KEY);
    if (saved) {
      predictionHistory = JSON.parse(saved);
    }
  } catch (e) {
    console.warn('Could not load history:', e);
    predictionHistory = [];
  }
}

function saveHistory() {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(predictionHistory));
  } catch (e) {
    console.warn('Could not save history:', e);
  }
}

function addToHistory(imageData, result) {
  const entry = {
    id: Date.now(),
    image: imageData,
    result: result,
    timestamp: new Date().toISOString(),
  };
  
  predictionHistory.unshift(entry);
  if (predictionHistory.length > MAX_HISTORY) {
    predictionHistory.pop();
  }
  saveHistory();
  renderHistory();
}

function clearHistory() {
  predictionHistory = [];
  saveHistory();
  renderHistory();
}

function renderHistory() {
  const container = $('#history-container');
  if (!container) return;
  
  if (predictionHistory.length === 0) {
    container.innerHTML = '<p class="history-empty">No predictions yet</p>';
    return;
  }
  
  container.innerHTML = predictionHistory.map(entry => {
    const r = entry.result;
    const color = r.color || CATEGORY_COLORS[r.class_name] || '#6b9b6b';
    const icon = CATEGORY_ICONS[r.class_name] || '♻️';
    const time = new Date(entry.timestamp).toLocaleTimeString();
    
    return `
      <div class="history-item" data-id="${entry.id}">
        <img src="${entry.image}" alt="Previous prediction" class="history-thumb">
        <div class="history-info">
          <span class="history-icon">${icon}</span>
          <span class="history-name">${r.display_name}</span>
          <span class="history-time">${time}</span>
        </div>
        <div class="history-confidence" style="background: ${color}">${(r.confidence * 100).toFixed(0)}%</div>
      </div>
    `;
  }).join('');
}

// ========== Tabs ==========
$$('.tab').forEach(tab => {
  tab.addEventListener('click', () => switchTab(tab.dataset.tab));
});

function switchTab(tab) {
  currentTab = tab;
  $$('.tab').forEach(t => {
    t.classList.toggle('tab-active', t.dataset.tab === tab);
    t.setAttribute('aria-selected', t.dataset.tab === tab);
  });

  // Hide all panels
  $('#panel-photo')?.classList.add('hidden');
  $('#panel-batch')?.classList.add('hidden');
  
  // Show selected panel
  if (tab === 'photo' || tab === 'upload') {
    $('#panel-photo')?.classList.remove('hidden');
  } else if (tab === 'batch') {
    $('#panel-batch')?.classList.remove('hidden');
  } else if (tab === 'guide') {
    document.querySelector('.guide-section')?.scrollIntoView({ behavior: 'smooth' });
    setTimeout(() => { switchTab('photo'); }, 100);
  }
}

// ========== Event Listeners ==========
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length > 0) {
    handleFile(e.dataTransfer.files[0]);
  }
});

dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

btnBrowse.addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.click();
});

btnCamera.addEventListener('click', (e) => {
  e.stopPropagation();
  cameraInput.click();
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

cameraInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

clearBtn.addEventListener('click', clearSelection);
classifyBtn.addEventListener('click', classifyImage);

// ========== Batch Event Listeners ==========
batchDropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  batchDropZone.classList.add('drag-over');
});

batchDropZone.addEventListener('dragleave', () => {
  batchDropZone.classList.remove('drag-over');
});

batchDropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  batchDropZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length > 0) {
    handleBatchFiles(e.dataTransfer.files);
  }
});

batchDropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    batchFileInput.click();
  }
});

btnBatchBrowse.addEventListener('click', (e) => {
  e.stopPropagation();
  batchFileInput.click();
});

batchFileInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) {
    handleBatchFiles(e.target.files);
  }
});

btnClearBatch.addEventListener('click', clearBatch);
btnClassifyBatch.addEventListener('click', classifyBatch);
btnClearBatchResults.addEventListener('click', clearBatchResults);

// ========== File Handling ==========
function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    showError('Please select an image file.');
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    dropZone.closest('.tab-panel').classList.add('hidden');
    previewContainer.classList.remove('hidden');
    classifyBtn.disabled = false;
    hideError();
    hideResults();
  };
  reader.readAsDataURL(file);
}

function clearSelection() {
  selectedFile = null;
  fileInput.value = '';
  cameraInput.value = '';
  previewImage.src = '';
  dropZone.closest('.tab-panel').classList.remove('hidden');
  previewContainer.classList.add('hidden');
  classifyBtn.disabled = true;
  hideResults();
  hideError();
}

// ========== Batch File Handling ==========
function handleBatchFiles(files) {
  const filesArray = Array.from(files).filter(f => f.type.startsWith('image/'));
  
  if (filesArray.length === 0) {
    showError('Please select image files.');
    return;
  }
  
  const remainingSlots = MAX_BATCH_FILES - batchFiles.length;
  if (remainingSlots <= 0) {
    showError(`Maximum ${MAX_BATCH_FILES} files allowed.`);
    return;
  }
  
  const toAdd = filesArray.slice(0, remainingSlots);
  const overflow = filesArray.length > remainingSlots;
  
  toAdd.forEach(file => {
    const id = 'batch-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    const reader = new FileReader();
    reader.onload = (e) => {
      batchFiles.push({
        id,
        file,
        preview: e.target.result
      });
      renderBatchFiles();
    };
    reader.readAsDataURL(file);
  });
  
  if (overflow) {
    showError(`Only added ${remainingSlots} files. Maximum ${MAX_BATCH_FILES} files allowed.`);
  }
  
  batchDropZone.classList.add('hidden');
  batchFileList.classList.remove('hidden');
}

function renderBatchFiles() {
  batchCount.textContent = `${batchFiles.length} file${batchFiles.length !== 1 ? 's' : ''} selected`;
  
  batchFilesContainer.innerHTML = batchFiles.map((item, index) => `
    <div class="batch-file-item" data-id="${item.id}">
      <img src="${item.preview}" alt="" class="batch-file-thumb">
      <div class="batch-file-info">
        <div class="batch-file-name">${item.file.name}</div>
        <div class="batch-file-size">${formatFileSize(item.file.size)}</div>
      </div>
      <button class="batch-file-remove" data-id="${item.id}" aria-label="Remove file">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"/>
          <line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
    </div>
  `).join('');
  
  // Add remove handlers
  batchFilesContainer.querySelectorAll('.batch-file-remove').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const id = btn.dataset.id;
      removeBatchFile(id);
    });
  });
  
  btnClassifyBatch.disabled = batchFiles.length === 0 || isBatchProcessing;
}

function removeBatchFile(id) {
  batchFiles = batchFiles.filter(f => f.id !== id);
  renderBatchFiles();
  
  if (batchFiles.length === 0) {
    clearBatch();
  }
}

function clearBatch() {
  batchFiles = [];
  batchFileInput.value = '';
  batchFileList.classList.add('hidden');
  batchDropZone.classList.remove('hidden');
  batchProgress.classList.add('hidden');
  btnClassifyBatch.disabled = true;
  hideError();
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ========== Batch Classification ==========
async function classifyBatch() {
  if (batchFiles.length === 0 || isBatchProcessing) return;
  
  isBatchProcessing = true;
  btnClassifyBatch.disabled = true;
  btnBatchContent.classList.add('hidden');
  btnBatchLoader.classList.remove('hidden');
  batchProgress.classList.remove('hidden');
  batchResults.classList.add('hidden');
  hideError();
  
  const formData = new FormData();
  batchFiles.forEach(item => {
    formData.append('files', item.file);
  });
  
  try {
    const resp = await fetch('/predict/batch', {
      method: 'POST',
      body: formData
    });
    
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${resp.status})`);
    }
    
    const results = await resp.json();
    displayBatchResults(results);
    
    // Add successful predictions to history (first one only to avoid spam)
    if (results.length > 0 && results[0].success && results[0].result) {
      addToHistory(batchFiles[0].preview, results[0].result);
    }
  } catch (err) {
    showError(err.message);
  } finally {
    isBatchProcessing = false;
    btnBatchContent.classList.remove('hidden');
    btnBatchLoader.classList.add('hidden');
    btnClassifyBatch.disabled = batchFiles.length === 0;
    batchProgress.classList.add('hidden');
  }
}

function displayBatchResults(results) {
  batchResultsList.innerHTML = '';
  
  results.forEach((item, index) => {
    const fileItem = batchFiles[index];
    const card = document.createElement('div');
    card.className = 'batch-result-card';
    card.style.animationDelay = `${index * 0.1}s`;
    
    if (!item.success || !item.result) {
      card.innerHTML = `
        <img src="${fileItem?.preview || ''}" alt="" class="batch-result-thumb">
        <div class="batch-result-info">
          <div class="batch-error">Failed: ${item.error || 'Unknown error'}</div>
          <div class="batch-file-name">${fileItem?.file?.name || 'Unknown'}</div>
        </div>
      `;
    } else {
      const r = item.result;
      const color = r.color || CATEGORY_COLORS[r.class_name] || '#6b9b6b';
      const icon = CATEGORY_ICONS[r.class_name] || '♻️';
      const pct = (r.confidence * 100).toFixed(1);
      
      card.innerHTML = `
        <img src="${fileItem?.preview || ''}" alt="" class="batch-result-thumb">
        <div class="batch-result-info">
          <div class="batch-result-header">
            <span class="batch-result-icon">${icon}</span>
            <span class="batch-result-class">${r.display_name}</span>
            <span class="batch-result-confidence" style="background: ${color}">${pct}%</span>
          </div>
          <div class="batch-result-desc">${r.description || ''}</div>
          <div class="batch-result-bar">
            <div class="batch-result-bar-fill" style="width: ${pct}%; background: ${color}"></div>
          </div>
        </div>
      `;
    }
    
    batchResultsList.appendChild(card);
  });
  
  batchResults.classList.remove('hidden');
  batchResults.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function clearBatchResults() {
  batchResults.classList.add('hidden');
  batchResultsList.innerHTML = '';
}

function updateBatchProgress(current, total) {
  const pct = (current / total) * 100;
  batchProgressFill.style.width = `${pct}%`;
  batchProgressText.textContent = `${current} / ${total}`;
  batchProgressLabel.textContent = `Processing ${current}/${total}...`;
}

// ========== Classification ==========
async function classifyImage() {
  if (!selectedFile) return;
  setLoading(true);
  hideError();
  hideResults();

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: formData });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${resp.status})`);
    }
    const result = await resp.json();
    displayResults(result);
    
    // Save to history
    const imageData = previewImage.src;
    addToHistory(imageData, result);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
}

function setLoading(on) {
  classifyBtn.disabled = on;
  btnContent.classList.toggle('hidden', on);
  btnLoader.classList.toggle('hidden', !on);
}

// ========== Results ==========
function displayResults(result) {
  const resultIcon = $('#result-icon');
  const resultIconBg = $('#result-icon-bg');
  const resultClass = $('#result-class');
  const resultDesc = $('#result-description');

  const color = result.color || CATEGORY_COLORS[result.class_name] || '#6b9b6b';
  const icon = CATEGORY_ICONS[result.class_name] || '♻️';

  resultIcon.textContent = icon;
  resultIconBg.style.background = `${color}20`;
  resultIconBg.style.borderColor = `${color}50`;

  resultClass.textContent = result.display_name;
  resultDesc.textContent = result.description;

  // Confidence
  const pct = (result.confidence * 100).toFixed(1);
  $('#confidence-fill').style.width = `${pct}%`;
  $('#confidence-fill').style.background = color;
  $('#confidence-value').textContent = `${pct}%`;

  // Confidence indicator class
  const confidenceLevel = result.confidence >= 0.8 ? 'high' : result.confidence >= 0.5 ? 'medium' : 'low';
  resultContainer.className = `result-panel confidence-${confidenceLevel}`;

  // Sorting tips
  const tipsContainer = $('#sorting-tips-container');
  const tipsList = $('#sorting-tips-list');
  const tips = SORTING_TIPS[result.class_name];
  if (tips && tips.length) {
    tipsList.innerHTML = tips.map((t) => `<li>${t}</li>`).join('');
    tipsContainer.classList.remove('hidden');
  } else {
    tipsContainer.classList.add('hidden');
  }

  // All probabilities
  const probList = $('#probabilities-list');
  probList.innerHTML = '';
  const sorted = Object.entries(result.all_probabilities).sort((a, b) => b[1] - a[1]);
  for (const [name, prob] of sorted) {
    const p = (prob * 100).toFixed(1);
    const catColor = CATEGORY_COLORS[name] || '#64748b';
    const el = document.createElement('div');
    el.className = 'category-bar-item';
    el.innerHTML = `
      <span class="category-bar-name">${name.replace(/_/g, ' ')}</span>
      <span class="category-bar-track">
        <span class="category-bar-fill" style="width:${p}%;background:${catColor}"></span>
      </span>
      <span class="category-bar-value">${p}%</span>
    `;
    probList.appendChild(el);
  }

  resultContainer.classList.remove('hidden');
  resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideResults() {
  resultContainer.classList.add('hidden');
}

// ========== Errors ==========
function showError(msg) {
  errorMessage.textContent = msg;
  errorContainer.classList.remove('hidden');
  setTimeout(() => {
    errorContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 10);
}

function hideError() {
  errorContainer.classList.add('hidden');
}

// ========== Category Cards ==========
function renderCategoryCards() {
  const grid = $('#categories-grid');
  const categories = [
    {
      icon: '♻️',
      name: 'Blue Bin',
      sub: 'Recyclables',
      color: CATEGORY_COLORS.blue_bin,
      desc: 'Cardboard, paper, plastic, metal, glass'
    },
    {
      icon: '🌿',
      name: 'Green Bin',
      sub: 'Organics',
      color: CATEGORY_COLORS.green_bin,
      desc: 'Food scraps, soiled paper, coffee grounds'
    },
    {
      icon: '🗑️',
      name: 'Garbage',
      sub: 'Black Bin',
      color: CATEGORY_COLORS.garbage,
      desc: 'Non-recyclables, compostable plastics'
    },
    {
      icon: '⚠️',
      name: 'Hazardous',
      sub: 'Household',
      color: CATEGORY_COLORS.hazardous,
      desc: 'Batteries, paint, chemicals, propane'
    },
    {
      icon: '💻',
      name: 'E-Waste',
      sub: 'Electronic',
      color: CATEGORY_COLORS.e_waste,
      desc: 'Computers, phones, cables'
    },
    {
      icon: '🍂',
      name: 'Yard Waste',
      sub: 'Organic',
      color: CATEGORY_COLORS.yard_waste,
      desc: 'Leaves, grass, branches'
    },
  ];

  grid.innerHTML = categories.map((c) => `
    <div class="category-card" style="color: ${c.color}">
      <div class="category-card-icon">${c.icon}</div>
      <div class="category-card-name">
        <span class="category-card-dot" style="background: ${c.color}"></span>
        ${c.name}
      </div>
      <div class="category-card-desc">${c.desc}</div>
    </div>
  `).join('');
}

// ========== Init ==========
function init() {
  initTheme();
  renderCategoryCards();
  loadHistory();
  renderHistory();
}


// ========== Theme Management ==========
function initTheme() {
  const saved = localStorage.getItem('ecosort-theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
  updateThemeIcon(saved);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('ecosort-theme', next);
  updateThemeIcon(next);
}

function updateThemeIcon(theme) {
  const btn = $('#theme-toggle');
  if (btn) btn.textContent = theme === 'dark' ? '🌙' : '☀️';
}

$('#theme-toggle')?.addEventListener('click', toggleTheme);

// Run init when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
