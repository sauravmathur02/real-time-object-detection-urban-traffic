document.addEventListener('DOMContentLoaded', () => {
    const slider = document.getElementById('conf-slider');
    const confVal = document.getElementById('conf-val');
    const fpsVal = document.getElementById('fps-val');
    const detectionsList = document.getElementById('detections-list');
    const sourceSelect = document.getElementById('source-select');
    const switchBtn = document.getElementById('switch-source-btn');
    
    // Sliders
    slider.addEventListener('input', (e) => {
        confVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    slider.addEventListener('change', async (e) => {
        const value = parseFloat(e.target.value);
        try {
            await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ conf_threshold: value })
            });
        } catch (err) {
            console.error("Error setting config:", err);
        }
    });

    // Source Switch
    switchBtn.addEventListener('click', async () => {
        const newSource = sourceSelect.value;
        try {
            switchBtn.textContent = "Switching...";
            await fetch('/api/source', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source: newSource })
            });
            switchBtn.textContent = "Switch Stream";
            // reload video stream
            const video = document.getElementById('video-stream');
            video.src = "/video_feed?" + new Date().getTime();
        } catch(err) {
            console.error("Error switching source", err);
            switchBtn.textContent = "Switch Stream";
        }
    });

    // Custom Video Upload
    const videoUpload = document.getElementById('video-upload');
    const uploadStatus = document.getElementById('upload-status');
    videoUpload.addEventListener('change', async (e) => {
        if (!e.target.files.length) return;
        const file = e.target.files[0];
        const formData = new FormData();
        formData.append('file', file);

        uploadStatus.textContent = "Uploading & processing...";
        uploadStatus.style.color = "#3b82f6";

        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.status === "success") {
                uploadStatus.textContent = "Upload complete! Stream updated.";
                uploadStatus.style.color = "#10b981";
                // reload video stream
                const video = document.getElementById('video-stream');
                video.src = "/video_feed?" + new Date().getTime();
            } else {
                uploadStatus.textContent = "Upload failed.";
                uploadStatus.style.color = "#ef4444";
            }
        } catch (err) {
            console.error("Upload error:", err);
            uploadStatus.textContent = "Upload failed.";
            uploadStatus.style.color = "#ef4444";
        }
    });


    // Poll Stats
    setInterval(async () => {
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();
            
            fpsVal.textContent = data.fps;
            
            // Render detections
            detectionsList.innerHTML = '';
            for (const [cls, count] of Object.entries(data.detections || {})) {
                if (count > 0) {
                    const li = document.createElement('li');
                    li.innerHTML = `<span>${cls}</span> <span class="cls-badge">${count}</span>`;
                    detectionsList.appendChild(li);
                }
            }
            if (detectionsList.innerHTML === '') {
                detectionsList.innerHTML = '<li><span style="color: #64748b">No objects detected</span></li>';
            }
        } catch (err) {
            // Server might be down
        }
    }, 500);
});
