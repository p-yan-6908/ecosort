const API_URL = '';

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');
const clearBtn = document.getElementById('clear-btn');
const classifyBtn = document.getElementById('classify-btn');
const btnText = document.getElementById('btn-text');
const btnLoader = document.getElementById('btn-loader');
const resultContainer = document.getElementById('result-container');
const errorContainer = document.getElementById('error-container');
const errorMessage = document.getElementById('error-message');

let selectedFile = null;

const categoryColors = {
    blue_bin: '#2563EB',
    green_bin: '#16A34A',
    garbage: '#1F2937',
    hazardous: '#DC2626',
    e_waste: '#7C3AED',
    yard_waste: '#65A30D'
};

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
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

clearBtn.addEventListener('click', clearSelection);
classifyBtn.addEventListener('click', classifyImage);

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file');
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        dropZone.classList.add('hidden');
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
    previewImage.src = '';
    dropZone.classList.remove('hidden');
    previewContainer.classList.add('hidden');
    classifyBtn.disabled = true;
    hideResults();
    hideError();
}

async function classifyImage() {
    if (!selectedFile) return;

    setLoading(true);
    hideError();
    hideResults();

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Classification failed');
        }

        const result = await response.json();
        displayResults(result);
    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(false);
    }
}

function setLoading(loading) {
    classifyBtn.disabled = loading;
    btnText.classList.toggle('hidden', loading);
    btnLoader.classList.toggle('hidden', !loading);
}

function displayResults(result) {
    document.getElementById('result-icon').textContent = result.icon;
    document.getElementById('result-class').textContent = result.display_name;
    document.getElementById('result-description').textContent = result.description;

    const confidencePercent = (result.confidence * 100).toFixed(1);
    document.getElementById('confidence-fill').style.width = `${confidencePercent}%`;
    document.getElementById('confidence-fill').style.background = result.color;
    document.getElementById('confidence-value').textContent = `${confidencePercent}%`;

    const probList = document.getElementById('probabilities-list');
    probList.innerHTML = '';

    const sorted = Object.entries(result.all_probabilities).sort((a, b) => b[1] - a[1]);
    for (const [className, prob] of sorted) {
        const percent = (prob * 100).toFixed(1);
        const color = categoryColors[className] || '#6B7280';

        const item = document.createElement('div');
        item.className = 'prob-item';
        item.innerHTML = `
            <span class="prob-bar">
                <span class="prob-fill" style="width: ${percent}%; background: ${color}"></span>
            </span>
            <span class="prob-name">${className.replace('_', ' ')}</span>
            <span class="prob-value">${percent}%</span>
        `;
        probList.appendChild(item);
    }

    resultContainer.classList.remove('hidden');
}

function hideResults() {
    resultContainer.classList.add('hidden');
}

function showError(message) {
    errorMessage.textContent = message;
    errorContainer.classList.remove('hidden');
}

function hideError() {
    errorContainer.classList.add('hidden');
}