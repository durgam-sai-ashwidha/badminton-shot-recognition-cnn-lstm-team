const videoInput = document.getElementById("videoInput");
const videoPreview = document.getElementById("videoPreview");
const noVideo = document.getElementById("noVideo");
const analyzeBtn = document.getElementById("analyzeBtn");


// Video upload
videoInput.addEventListener("change", function () {

    const file = this.files[0];

    if (!file) {
        return;
    }

    const videoURL = URL.createObjectURL(file);

    videoPreview.src = videoURL;
    videoPreview.style.display = "block";
    noVideo.style.display = "none";

    console.log("Selected video:", file.name);
});


// Analyze button
analyzeBtn.addEventListener("click", async function () {

    const file = videoInput.files[0];

    if (!file) {
        alert("Please upload a video first.");
        return;
    }

    analyzeBtn.innerText = "⏳ Analyzing...";
    analyzeBtn.disabled = true;

    try {
        const formData = new FormData();
        formData.append("video", file);

        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            let errorMsg = `Server error (${response.status})`;
            try {
                const errJson = await response.json();
                if (errJson && errJson.detail) {
                    errorMsg = errJson.detail;
                }
            } catch (_) {}
            throw new Error(errorMsg);
        }

        const data = await response.json();

        // Update primary prediction display
        document.getElementById("prediction").innerText = (data.predicted_class || "---").toUpperCase();

        const topProb = Math.round((data.probabilities[data.predicted_class] || 0) * 100);
        document.getElementById("confidence").innerText = `${topProb}%`;

        // Update class probabilities breakdown
        const probs = data.probabilities || {};
        document.getElementById("smash").innerText = `${Math.round((probs["Smash"] || 0) * 100)}%`;
        document.getElementById("clear").innerText = `${Math.round((probs["Clear"] || 0) * 100)}%`;
        document.getElementById("drop").innerText = `${Math.round((probs["Drop"] || 0) * 100)}%`;
        document.getElementById("drive").innerText = `${Math.round((probs["Drive"] || 0) * 100)}%`;
        document.getElementById("net").innerText = `${Math.round((probs["Net Shot"] || 0) * 100)}%`;

    } catch (error) {
        alert(`Analysis failed: ${error.message}`);
    } finally {
        analyzeBtn.innerText = "▶ Analyze Video";
        analyzeBtn.disabled = false;
    }

});