// ========== State ==========
let selectedFile = null;
let currentTab = 'photo';

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

  // Show/hide panels
  const panels = {
    'photo': $('#panel-photo'),
    'upload': $('#panel-photo'), // Same panel
    'guide': null
  };

  // Scroll to guide section if guide tab
  if (tab === 'guide') {
    document.querySelector('.guide-section')?.scrollIntoView({ behavior: 'smooth' });
    // Switch back to photo tab visually
    setTimeout(() => {
      switchTab('photo');
    }, 100);
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

// Tab "keyboard" navigation
dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

// Browse button - separate from drop zone
btnBrowse.addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.click();
});

// Camera button
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
    displayResults(await resp.json());
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

  // Use returned HTML color or fallback
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
renderCategoryCards();
