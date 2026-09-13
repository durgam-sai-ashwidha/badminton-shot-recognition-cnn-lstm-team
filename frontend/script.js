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


    // TEMPORARY DEMO RESULT
    // Later this will come from our Python backend.

    setTimeout(() => {

        document.getElementById("prediction").innerText = "SMASH";
        document.getElementById("confidence").innerText = "84%";

        document.getElementById("smash").innerText = "84%";
        document.getElementById("clear").innerText = "6%";
        document.getElementById("drop").innerText = "4%";
        document.getElementById("drive").innerText = "3%";
        document.getElementById("net").innerText = "3%";

        analyzeBtn.innerText = "✓ Analysis Complete";
        analyzeBtn.disabled = false;

    }, 1500);

});