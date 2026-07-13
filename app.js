// App Controller - Dynamic Rendering and Local Admin
document.addEventListener('DOMContentLoaded', () => {
    // Load state from window.WORKSHOP_DATA
    let data = window.WORKSHOP_DATA || { info: {}, speakers: [], schedule: [], news: [], organizers: [] };

    // DOM Elements
    const heroTitle = document.getElementById('hero-title');
    const heroSubtitle = document.getElementById('hero-subtitle');
    const heroDate = document.getElementById('hero-date');
    const heroLocation = document.getElementById('hero-location');
    const aboutTextP = document.getElementById('about-text-p');
    
    const speakersContainer = document.getElementById('speakers-container');
    const scheduleContainer = document.getElementById('schedule-container');
    const organizersContainer = document.getElementById('organizers-container');
    const dropdownNewsList = document.getElementById('dropdown-news-list');
    
    // Modals
    const speakerBioModal = new bootstrap.Modal(document.getElementById('speakerBioModal'));
    const bioModalBody = document.getElementById('bioModalBody');
    const bioModalLabel = document.getElementById('bioModalLabel');

    // Admin elements
    const uploadSlotSelect = document.getElementById('upload-slot-select');
    const slideDropzone = document.getElementById('slide-dropzone');
    const slideFileInput = document.getElementById('slide-file-input');
    const uploadFileStatus = document.getElementById('upload-file-status');
    const adminNewsList = document.getElementById('admin-news-list');

    // Current pending slide file name
    let pendingSlideFile = null;

    // 1. Initial Render
    function renderPage() {
        // Render Workshop Info
        if (data.info) {
            heroTitle.textContent = data.info.title || 'AI for Psychiatry';
            heroSubtitle.textContent = data.info.subtitle || '';
            heroDate.textContent = data.info.date || '';
            heroLocation.textContent = data.info.location || '';
            aboutTextP.textContent = data.info.description || '';
        }

        // Render Speakers
        renderSpeakers();

        // Render Schedule
        renderSchedule();

        // Render Organizers
        renderOrganizers();

        // Render News
        renderNews();

        // Populate admin talk slots selection list
        populateAdminTalks();
    }

    // Render Speakers List
    function renderSpeakers() {
        speakersContainer.innerHTML = '';
        if (!data.speakers || data.speakers.length === 0) {
            speakersContainer.innerHTML = '<div class="text-center text-muted col-12">No speakers listed yet.</div>';
            return;
        }

        data.speakers.forEach(speaker => {
            const card = document.createElement('div');
            card.className = 'glass-panel speaker-card';
            
            // Social icons block
            let socialsHTML = '';
            if (speaker.socials) {
                if (speaker.socials.website) socialsHTML += `<a href="${speaker.socials.website}" target="_blank" title="Website"><i class="fas fa-globe"></i></a>`;
                if (speaker.socials.twitter) socialsHTML += `<a href="${speaker.socials.twitter}" target="_blank" title="Twitter/X"><i class="fab fa-twitter"></i></a>`;
                if (speaker.socials.linkedin) socialsHTML += `<a href="${speaker.socials.linkedin}" target="_blank" title="LinkedIn"><i class="fab fa-linkedin"></i></a>`;
            }

            const imgHTML = speaker.image 
                ? `<img src="${speaker.image}" alt="${speaker.name}" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                   <div class="speaker-img-placeholder" style="display: none; height: 100%; width: 100%; align-items: center; justify-content: center;"><i class="fas fa-user-md"></i></div>`
                : `<div class="speaker-img-placeholder"><i class="fas fa-user-md"></i></div>`;

            card.innerHTML = `
                <div class="speaker-img-container">
                    ${imgHTML}
                </div>
                <h4 class="speaker-name">${speaker.name}</h4>
                <div class="speaker-title">${speaker.title}</div>
                <div class="speaker-institution">${speaker.institution}</div>
                <div class="social-icons" onclick="event.stopPropagation()">
                    ${socialsHTML}
                </div>
            `;

            // Open biography modal on click
            card.addEventListener('click', () => {
                showSpeakerBio(speaker);
            });

            speakersContainer.appendChild(card);
        });
    }

    // Show Speaker Biography Detail Modal
    function showSpeakerBio(speaker) {
        bioModalLabel.textContent = `${speaker.name} - Biography`;
        
        let socialsHTML = '';
        if (speaker.socials) {
            if (speaker.socials.website) socialsHTML += `<a href="${speaker.socials.website}" target="_blank" class="btn btn-sm btn-outline-light me-2"><i class="fas fa-globe me-1"></i> Website</a>`;
            if (speaker.socials.twitter) socialsHTML += `<a href="${speaker.socials.twitter}" target="_blank" class="btn btn-sm btn-outline-light me-2"><i class="fab fa-twitter me-1"></i> Twitter</a>`;
            if (speaker.socials.linkedin) socialsHTML += `<a href="${speaker.socials.linkedin}" target="_blank" class="btn btn-sm btn-outline-light me-2"><i class="fab fa-linkedin me-1"></i> LinkedIn</a>`;
        }

        const modalImgHTML = speaker.image 
            ? `<img src="${speaker.image}" alt="${speaker.name}" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
               <div class="speaker-img-placeholder" style="display: none; height: 100%; width: 100%; align-items: center; justify-content: center;"><i class="fas fa-user-md" style="font-size: 2.2rem;"></i></div>`
            : `<div class="speaker-img-placeholder" style="display: flex; height: 100%; width: 100%; align-items: center; justify-content: center;"><i class="fas fa-user-md" style="font-size: 2.2rem;"></i></div>`;

        bioModalBody.innerHTML = `
            <div class="row align-items-center mb-4">
                <div class="col-md-3 text-center mb-3 mb-md-0">
                    <div class="speaker-img-container m-0">
                        ${modalImgHTML}
                    </div>
                </div>
                <div class="col-md-9">
                    <h3>${speaker.name}</h3>
                    <div class="text-teal fw-bold mb-1" style="color: var(--color-teal);">${speaker.title}</div>
                    <div class="text-secondary">${speaker.institution}</div>
                </div>
            </div>
            <div class="bio-text mb-4" style="line-height: 1.6; font-size: 1.05rem; color: var(--text-secondary);">
                ${speaker.bio || 'Biography placeholder.'}
            </div>
            <div class="border-top pt-3 text-end">
                ${socialsHTML}
            </div>
        `;
        speakerBioModal.show();
    }

    // Render Schedule List
    function renderSchedule() {
        scheduleContainer.innerHTML = '';
        if (!data.schedule || data.schedule.length === 0) {
            scheduleContainer.innerHTML = '<div class="text-center text-muted col-12">No schedule items added yet.</div>';
            return;
        }

        data.schedule.forEach(item => {
            const card = document.createElement('div');
            card.className = 'glass-panel schedule-card';

            const catClass = getCategoryClass(item.category);
            
            // Check slide status
            const hasSlides = item.slidesUrl && item.slidesUrl.trim() !== "";
            const btnLabel = hasSlides ? '<i class="fas fa-download me-1"></i> Slides' : 'Slides Unavailable';
            const btnClass = hasSlides ? 'btn-premium' : 'btn-premium disabled';
            const downloadAttr = hasSlides ? `href="${item.slidesUrl}" download` : '';

            card.innerHTML = `
                <div class="schedule-time-block">
                    <div class="schedule-time">${item.time}</div>
                    <span class="schedule-category ${catClass}">${item.category || 'Event'}</span>
                </div>
                <div class="schedule-info-block">
                    <div>
                        <h4 class="schedule-title">${item.title}</h4>
                        ${item.speaker ? `<div class="schedule-speaker"><i class="far fa-user me-1"></i> ${item.speaker}</div>` : ''}
                        <p class="schedule-desc mb-0">${item.description || ''}</p>
                    </div>
                    <div class="schedule-action-block">
                        <a ${downloadAttr} class="${btnClass}">${btnLabel}</a>
                    </div>
                </div>
            `;
            scheduleContainer.appendChild(card);
        });
    }

    function getCategoryClass(category) {
        if (!category) return 'cat-break';
        const cat = category.toLowerCase();
        if (cat.includes('keynote') || cat.includes('talk') || cat.includes('presentation')) return 'cat-keynote';
        if (cat.includes('opening') || cat.includes('closing') || cat.includes('intro')) return 'cat-opening';
        if (cat.includes('break') || cat.includes('coffee') || cat.includes('lunch')) return 'cat-break';
        if (cat.includes('panel') || cat.includes('discussion')) return 'cat-panel';
        return 'cat-break';
    }

    // Render Organizers List
    function renderOrganizers() {
        organizersContainer.innerHTML = '';
        if (!data.organizers || data.organizers.length === 0) {
            organizersContainer.innerHTML = '<div class="text-center text-muted col-12">No organizers listed yet.</div>';
            return;
        }

        data.organizers.forEach(org => {
            const card = document.createElement('div');
            card.className = 'glass-panel organizer-card';

            let socialsHTML = '';
            if (org.socials) {
                if (org.socials.website) socialsHTML += `<a href="${org.socials.website}" target="_blank" title="Website"><i class="fas fa-globe"></i></a>`;
                if (org.socials.twitter) socialsHTML += `<a href="${org.socials.twitter}" target="_blank" title="Twitter/X"><i class="fab fa-twitter"></i></a>`;
                if (org.socials.linkedin) socialsHTML += `<a href="${org.socials.linkedin}" target="_blank" title="LinkedIn"><i class="fab fa-linkedin"></i></a>`;
            }

            const imgHTML = org.image 
                ? `<img src="${org.image}" alt="${org.name}" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                   <div class="organizer-img-placeholder" style="display: none; height: 100%; width: 100%; align-items: center; justify-content: center;"><i class="fas fa-user-cog"></i></div>`
                : `<div class="organizer-img-placeholder"><i class="fas fa-user-cog"></i></div>`;

            card.innerHTML = `
                <div class="organizer-img-container">
                    ${imgHTML}
                </div>
                <div class="organizer-details">
                    <h5 class="organizer-name">${org.name}</h5>
                    <div class="organizer-title">${org.title}</div>
                    <div class="organizer-institution">${org.institution}</div>
                    <div class="social-icons justify-content-start mt-2">
                        ${socialsHTML}
                    </div>
                </div>
            `;
            organizersContainer.appendChild(card);
        });
    }

    // Render News Dropdown & Admin News List
    function renderNews() {
        dropdownNewsList.innerHTML = '';
        adminNewsList.innerHTML = '';

        if (!data.news || data.news.length === 0) {
            dropdownNewsList.innerHTML = '<div class="text-center text-muted py-2 small">No updates yet.</div>';
            adminNewsList.innerHTML = '<div class="text-center text-muted p-2 small">No updates yet.</div>';
            return;
        }

        data.news.forEach((post, index) => {
            // Format date for badge
            let month = 'AUG';
            let day = '20';
            if (post.date) {
                const dateObj = new Date(post.date);
                if (!isNaN(dateObj)) {
                    month = dateObj.toLocaleString('default', { month: 'short' }).toUpperCase();
                    day = dateObj.getDate();
                }
            }

            const tagClass = `tag-${(post.tag || 'update').toLowerCase()}`;

            // Dropdown HTML
            const dropItem = document.createElement('div');
            dropItem.className = 'update-item';
            dropItem.innerHTML = `
                <div class="update-date">
                    <span class="day">${day}</span>
                    <span class="month">${month}</span>
                </div>
                <div class="update-details">
                    <div class="update-header-info">
                        <span class="update-item-title">${post.title}</span>
                        <span class="update-tag ${tagClass}">${post.tag || 'Update'}</span>
                    </div>
                    <p class="update-text">${post.content}</p>
                </div>
            `;
            dropdownNewsList.appendChild(dropItem);

            // Admin List HTML
            const adminRow = document.createElement('div');
            adminRow.className = 'admin-news-row';
            adminRow.innerHTML = `
                <div>
                    <span class="badge ${tagClass} me-2">${post.tag || 'Update'}</span>
                    <strong>${post.title}</strong> <small class="text-muted">(${post.date})</small>
                </div>
                <button type="button" class="btn btn-sm btn-danger text-light py-0 px-2" onclick="deleteUpdate(${index})">
                    <i class="fas fa-trash-alt" style="font-size: 0.8rem;"></i>
                </button>
            `;
            adminNewsList.appendChild(adminRow);
        });
    }

    // Populate drop-down list in Admin panel for choosing schedule talk slots
    function populateAdminTalks() {
        uploadSlotSelect.innerHTML = '';
        if (!data.schedule || data.schedule.length === 0) return;

        data.schedule.forEach(item => {
            // Only make presentations/keynotes eligible for slides
            if (item.category && (item.category.toLowerCase().includes('keynote') || item.category.toLowerCase().includes('talk') || item.category.toLowerCase().includes('presentation') || item.category.toLowerCase().includes('panel'))) {
                const opt = document.createElement('option');
                opt.value = item.id;
                opt.textContent = `[${item.time}] ${item.title} (${item.speaker || 'No Speaker'})`;
                uploadSlotSelect.appendChild(opt);
            }
        });
    }

    // Drag and Drop Uploader Event Handlers
    ['dragenter', 'dragover'].forEach(eventName => {
        slideDropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            slideDropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        slideDropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            slideDropzone.classList.remove('dragover');
        }, false);
    });

    slideDropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleSlideFile(files[0]);
        }
    });

    slideFileInput.addEventListener('change', (e) => {
        if (slideFileInput.files.length > 0) {
            handleSlideFile(slideFileInput.files[0]);
        }
    });

    function handleSlideFile(file) {
        pendingSlideFile = file;
        
        // Visual indicator
        uploadFileStatus.classList.remove('d-none');
        uploadFileStatus.innerHTML = `<i class="fas fa-check-circle me-1"></i> File selected: <strong>${file.name}</strong> (Ready to Link)`;

        // Link slides immediately in state
        const selectedSlotId = uploadSlotSelect.value;
        const targetSlot = data.schedule.find(item => item.id === selectedSlotId);
        if (targetSlot) {
            // Assign slide path in slides directory
            targetSlot.slidesUrl = `slides/${file.name}`;
            
            // Re-render schedule to reflect changes locally
            renderSchedule();
        }
    }

    // Add news post
    window.addNewUpdate = function() {
        const titleInput = document.getElementById('news-title');
        const tagInput = document.getElementById('news-tag');
        const contentInput = document.getElementById('news-content');

        if (!titleInput.value.trim() || !contentInput.value.trim()) {
            alert('Please enter both a title and content details for the update.');
            return;
        }

        const newPost = {
            date: new Date().toISOString().split('T')[0], // yyyy-mm-dd
            tag: tagInput.value,
            title: titleInput.value.trim(),
            content: contentInput.value.trim()
        };

        // Add to beginning of news
        data.news.unshift(newPost);

        // Clear inputs
        titleInput.value = '';
        contentInput.value = '';

        // Re-render
        renderNews();
    };

    // Delete news post
    window.deleteUpdate = function(index) {
        if (confirm('Are you sure you want to delete this update?')) {
            data.news.splice(index, 1);
            renderNews();
        }
    };

    // Tab Switching in Admin Panel
    window.switchAdminTab = function(tabId, btn) {
        // Toggle tab buttons
        document.querySelectorAll('.admin-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Toggle panels
        document.querySelectorAll('.admin-panel-content').forEach(p => p.classList.remove('active'));
        document.getElementById(tabId).classList.add('active');
    };

    // Export config data.js file
    window.exportDataFile = function() {
        const fileContent = `// Workshop Config Data\nwindow.WORKSHOP_DATA = ${JSON.stringify(data, null, 2)};\n`;
        const blob = new Blob([fileContent], { type: 'application/javascript;charset=utf-8' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'data.js';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        alert('Configuration exported! Please place the downloaded "data.js" in your project workspace folder to save changes. Also put the slide file under the "slides" directory if you linked slide presentations.');
    };

    // Toggle Admin Panel modal via keyboard shortcut (Ctrl+M)
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key.toLowerCase() === 'm') {
            e.preventDefault();
            const adminModal = new bootstrap.Modal(document.getElementById('adminModal'));
            adminModal.toggle();
        }
    });

    // Populate slot select change tracking
    uploadSlotSelect.addEventListener('change', () => {
        // Reset file uploader status if slot is changed
        pendingSlideFile = null;
        uploadFileStatus.classList.add('d-none');
    });

    // Init Page Run
    renderPage();
});
