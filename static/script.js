const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const loadingState = document.getElementById('loadingState');
const resultCard = document.getElementById('resultCard');
const errorCard = document.getElementById('errorCard');

// Event Listeners
if (dropZone) {
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--primary)';
        dropZone.style.background = '#F0FDF4';
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#E2E8F0';
        dropZone.style.background = '#FFFFFF';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#E2E8F0';
        dropZone.style.background = '#FFFFFF';
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });
}

if (fileInput) {
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }
    uploadImage(file);
}

async function uploadImage(file) {
    // UI State
    dropZone.classList.add('hidden');
    errorCard.classList.add('hidden');
    resultCard.classList.add('hidden');
    loadingState.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        loadingState.classList.add('hidden');

        if (response.ok) {
            if (data.is_crop) {
                showResults(data);
            } else {
                showError(data.message || "This image doesn't look like a crop.");
            }
        } else {
            showError(data.error || 'Unknown error occurred');
        }

    } catch (error) {
        console.error('Error:', error);
        loadingState.classList.add('hidden');
        resultCard.classList.add('hidden'); // Ensure result is hidden if error occurs
        showError(error.message || 'Failed to connect to the server.');
    } finally {
        fileInput.value = '';
    }
}

function showResults(data) {
    const details = data.details;
    const diseaseName = details.disease_name || data.class_name;
    const confidence = (data.confidence * 100).toFixed(1);

    document.getElementById('resImage').src = data.image_url;
    document.getElementById('resName').innerText = diseaseName;
    document.getElementById('resConfidence').innerText = `${confidence}% Confidence`;

    // Lists
    populateList('symptomsList', details.symptoms);
    populateList('treatmentList', details.management_steps);
    populateList('preventionList', details.prevention);

    resultCard.classList.remove('hidden');

    // Default tab
    openTab('symptoms');
}

function showError(msg) {
    document.getElementById('errorMessage').innerText = msg;
    errorCard.classList.remove('hidden');
}

function resetDiagnosis() {
    resultCard.classList.add('hidden');
    errorCard.classList.add('hidden');
    dropZone.classList.remove('hidden');
}

function populateList(elementId, items) {
    const list = document.getElementById(elementId);
    list.innerHTML = '';
    if (items && items.length > 0) {
        items.forEach(item => {
            const li = document.createElement('li');
            li.innerText = item;
            list.appendChild(li);
        });
    } else {
        list.innerHTML = '<li>No specific information available.</li>';
    }
}

// TABS
// TABS
function openTab(tabName) {
    // Buttons
    document.querySelectorAll('.tab-link').forEach(btn => {
        btn.classList.remove('active');
        // Check if this button controls the requested tab
        if (btn.innerText.toLowerCase().includes(tabName) ||
            btn.getAttribute('onclick').includes(tabName)) {
            btn.classList.add('active');
        }
    });

    // Content
    document.querySelectorAll('.info-list').forEach(list => list.classList.remove('active-list'));

    let listId = '';
    if (tabName === 'symptoms') listId = 'symptomsList';
    if (tabName === 'treatment') listId = 'treatmentList';
    if (tabName === 'prevention') listId = 'preventionList';

    if (listId) {
        document.getElementById(listId).classList.add('active-list');
    }
}

// NAVIGATION & HISTORY
function showSection(sectionId) {
    // Nav Active State
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    // Ideally select the button clicked, but for now simple check
    if (sectionId === 'diagnosis') document.querySelector('.nav-btn:nth-child(1)').classList.add('active');
    if (sectionId === 'history') document.querySelector('.nav-btn:nth-child(2)').classList.add('active');

    // Sections
    document.querySelectorAll('.section').forEach(sec => sec.classList.remove('active'));
    document.getElementById(`${sectionId}-section`).classList.add('active');

    if (sectionId === 'history') {
        loadHistory();
    }
}

async function loadHistory() {
    const grid = document.getElementById('historyGrid');
    grid.innerHTML = '<div class="spinner"></div>';

    try {
        const response = await fetch('/api/history');
        const history = await response.json();

        grid.innerHTML = '';

        if (history.length === 0) {
            grid.innerHTML = `
                <div class="empty-state" style="grid-column: 1/-1; text-align: center; color: var(--text-secondary); padding: 3rem;">
                    <i class="ph ph-clock" style="font-size: 2rem; margin-bottom:1rem;"></i>
                    <p>No history yet.</p>
                </div>`;
            return;
        }

        history.forEach(item => {
            const date = new Date(item.date).toLocaleDateString();
            const confidence = (item.confidence * 100).toFixed(0);

            const card = document.createElement('div');
            card.className = 'history-card';

            let statusHtml = '';
            if (item.is_crop) {
                statusHtml = `<span class="history-tag success">${confidence}% Confidence</span>`;
            } else {
                statusHtml = `<span class="history-tag error">Not a Crop</span>`;
            }

            card.innerHTML = `
                <div class="history-img">
                    <img src="${item.image_url}" alt="History Image" loading="lazy">
                </div>
                <div class="history-content">
                    <div class="history-date">${date}</div>
                    <div class="history-title">${item.disease_name}</div>
                    ${statusHtml}
                </div>
            `;
            grid.appendChild(card);
        });

    } catch (e) {
        grid.innerText = 'Failed to load history.';
        console.error(e);
    }
}
