import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";
const ACCEPTED_TYPES = ".mp4,.avi,.mov,.mkv";
const CLASS_NAMES = ["Clear", "Drive", "Drop", "Net_Shot", "Smash"];

function Icon({ name, size = 20 }) {
  const paths = {
    settings: <><circle cx="12" cy="12" r="3" /><path d="m19.4 15 .1.1 1.4 1.1-1.4 2.4-1.7-.6a8 8 0 0 1-1.5.9l-.3 1.8h-2.8l-.3-1.8a8 8 0 0 1-1.5-.9l-1.7.6-1.4-2.4 1.4-1.1a7 7 0 0 1 0-1.8l-1.4-1.1 1.4-2.4 1.7.6a8 8 0 0 1 1.5-.9l.3-1.8h2.8l.3 1.8a8 8 0 0 1 1.5.9l1.7-.6 1.4 2.4-1.4 1.1a7 7 0 0 1 0 1.7Z" transform="translate(-2 -2) scale(1.16)" /></>,
    video: <><rect x="3" y="5" width="13" height="14" rx="2" /><path d="m16 10 5-3v10l-5-3z" /></>,
    layers: <><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 12 9 5 9-5M3 16l9 5 9-5" /></>,
    trophy: <><path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0V4Z" /><path d="M7 6H3v3a4 4 0 0 0 4 4M17 6h4v3a4 4 0 0 1-4 4" /></>,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 8h.01" /></>,
    help: <><circle cx="12" cy="12" r="9" /><path d="M9.6 9a2.5 2.5 0 1 1 4.3 1.7c-1.1 1-1.9 1.3-1.9 3M12 17h.01" /></>,
    upload: <><path d="M12 16V4m-5 5 5-5 5 5" /><path d="M4 16v4h16v-4" /></>,
    file: <><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9Z" /><path d="M13 2v7h7M10 13l5 3-5 3v-6Z" /></>,
    check: <><circle cx="12" cy="12" r="9" /><path d="m8 12 3 3 5-6" /></>,
    close: <path d="m18 6-12 12M6 6l12 12" />,
    arrow: <><path d="M7 17 17 7M7 7h10v10" /></>,
    chevron: <path d="m9 18 6-6-6-6" />,
    trash: <><path d="M3 6h18M8 6V4h8v2m-11 0 1 15h10l1-15M10 10v7m4-7v7" /></>,
    activity: <><path d="M3 12h4l3-8 4 16 3-8h4" /></>,
    alert: <><path d="M12 3 2.8 20h18.4L12 3Z" /><path d="M12 9v4m0 3h.01" /></>,
  };

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name] || paths.activity}
    </svg>
  );
}

function App() {
  const [mode, setMode] = useState("single");
  const [menuOpen, setMenuOpen] = useState(false);
  const [dialog, setDialog] = useState("");
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);
  const menuRef = useRef(null);

  useEffect(() => {
    function handleOutside(event) {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setMenuOpen(false);
      }
    }

    function handleEscape(event) {
      if (event.key === "Escape") {
        setMenuOpen(false);
        setDialog("");
      }
    }

    document.addEventListener("mousedown", handleOutside);
    document.addEventListener("keydown", handleEscape);

    return () => {
      document.removeEventListener("mousedown", handleOutside);
      document.removeEventListener("keydown", handleEscape);
    };
  }, []);

  function chooseMode(nextMode) {
    setMenuOpen(false);

    if (nextMode === "complete") {
      setDialog("coming-soon");
      return;
    }

    setMode(nextMode);
    setFiles([]);
    setResults([]);
    setError("");
    setProgress({ done: 0, total: 0 });
  }

  function addFiles(fileList) {
    const selected = Array.from(fileList || []);

    if (!selected.length) return;

    const valid = selected.filter((file) => {
      const extension = file.name.split(".").pop().toLowerCase();
      return ["mp4", "avi", "mov", "mkv"].includes(extension);
    });

    if (valid.length !== selected.length) {
      setError("Some files were skipped. Choose MP4, AVI, MOV, or MKV videos.");
    } else {
      setError("");
    }

    const combined = mode === "single" ? valid.slice(0, 1) : [...files, ...valid];
    const unique = combined.filter(
      (file, index, array) =>
        array.findIndex(
          (other) =>
            other.name === file.name &&
            other.size === file.size &&
            other.lastModified === file.lastModified
        ) === index
    );

    setFiles(unique);
    setResults([]);
  }

  function removeFile(index) {
    setFiles((current) => current.filter((_, i) => i !== index));
    setResults([]);
  }

  async function predictOne(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    const contentType = response.headers.get("content-type") || "";
    const data = contentType.includes("application/json")
      ? await response.json()
      : { detail: await response.text() };

    if (!response.ok) {
      throw new Error(data.detail || data.error || `Server error (${response.status})`);
    }

    const probabilities = data.probabilities || data.class_probabilities || {};
    let prediction =
      data.predicted_class ||
      data.prediction ||
      data.predicted_stroke ||
      data.stroke ||
      data.class_name;

    let confidence =
      data.confidence ??
      data.model_confidence ??
      data.confidence_score;

    if (typeof prediction === "object" && prediction !== null) {
      confidence = prediction.confidence ?? confidence;
      prediction = prediction.label || prediction.class_name || prediction.name;
    }

    if (!prediction && Object.keys(probabilities).length) {
      prediction = Object.entries(probabilities).sort(
        (a, b) => Number(b[1]) - Number(a[1])
      )[0][0];
    }

    if (!prediction) {
      throw new Error(
        "The backend responded, but its prediction field was not recognized. Check the JSON returned by /predict."
      );
    }

    const normalizedProbabilities = Object.entries(probabilities).map(
      ([label, value]) => ({
        label: label.replaceAll("_", " "),
        value: Number(value) > 1 ? Number(value) : Number(value) * 100,
      })
    );

    if (confidence == null && normalizedProbabilities.length) {
      confidence = Math.max(...normalizedProbabilities.map((item) => item.value));
    }

    confidence = Number(confidence ?? 0);
    if (confidence <= 1) confidence *= 100;

    return {
      filename: file.name,
      prediction: String(prediction).replaceAll("_", " "),
      confidence: Math.max(0, Math.min(100, confidence)),
      probabilities: normalizedProbabilities,
    };
  }

  async function handlePredict() {
    if (!files.length || loading) return;

    setLoading(true);
    setError("");
    setResults([]);
    setProgress({ done: 0, total: files.length });

    const completed = [];

    try {
      for (const file of files) {
        try {
          const result = await predictOne(file);
          completed.push({ ...result, status: "success" });
        } catch (err) {
          completed.push({
            filename: file.name,
            status: "error",
            message: err.message || "Prediction failed.",
          });
        }

        setResults([...completed]);
        setProgress({ done: completed.length, total: files.length });
      }

      if (completed.every((result) => result.status === "error")) {
        setError("No clips could be analyzed. Check that the backend is running and its /predict endpoint accepts the selected video.");
      }
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function resetAnalysis() {
    if (inputRef.current) inputRef.current.value = "";
    setFiles([]);
    setResults([]);
    setError("");
    setProgress({ done: 0, total: 0 });
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#home" aria-label="RallyAI home">
          <span className="brand-symbol"><Icon name="activity" size={23} /></span>
          <span>Rally<span className="brand-accent">AI</span></span>
        </a>

        <div className="header-right">
          <span className="connection-label">
            <span className="status-dot" />
            LOCAL ANALYSIS
          </span>

          <div className="settings-wrap" ref={menuRef}>
            <button
              className={`settings-button ${menuOpen ? "is-open" : ""}`}
              onClick={() => setMenuOpen((open) => !open)}
              aria-label="Open settings"
              aria-expanded={menuOpen}
              title="Settings"
            >
              <Icon name="settings" size={20} />
            </button>

            {menuOpen && (
              <div className="settings-menu">
                <div className="menu-heading">ANALYSIS MODE</div>

                <button
                  className={`menu-option ${mode === "single" ? "selected" : ""}`}
                  onClick={() => chooseMode("single")}
                >
                  <span className="menu-icon"><Icon name="video" /></span>
                  <span className="menu-copy">
                    <strong>Single clip</strong>
                    <small>Analyze one video</small>
                  </span>
                  {mode === "single" && <Icon name="check" size={18} />}
                </button>

                <button
                  className={`menu-option ${mode === "multiple" ? "selected" : ""}`}
                  onClick={() => chooseMode("multiple")}
                >
                  <span className="menu-icon"><Icon name="layers" /></span>
                  <span className="menu-copy">
                    <strong>Multiple clips</strong>
                    <small>Analyze several videos</small>
                  </span>
                  {mode === "multiple" && <Icon name="check" size={18} />}
                </button>

                <button
                  className="menu-option"
                  onClick={() => chooseMode("complete")}
                >
                  <span className="menu-icon muted"><Icon name="trophy" /></span>
                  <span className="menu-copy">
                    <strong>Complete match</strong>
                    <small>Not available yet</small>
                  </span>
                </button>

                <div className="menu-divider" />

                <button className="menu-option" onClick={() => { setMenuOpen(false); setDialog("about"); }}>
                  <span className="menu-icon"><Icon name="info" /></span>
                  <span className="menu-copy">
                    <strong>About RallyAI</strong>
                    <small>Project overview</small>
                  </span>
                  <Icon name="chevron" size={17} />
                </button>

                <button className="menu-option" onClick={() => { setMenuOpen(false); setDialog("help"); }}>
                  <span className="menu-icon"><Icon name="help" /></span>
                  <span className="menu-copy">
                    <strong>Help</strong>
                    <small>How to use the website</small>
                  </span>
                  <Icon name="chevron" size={17} />
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      <main id="home" className="main-content">
        <section className="hero">
          <div className="eyebrow"><span className="eyebrow-dot" /> BADMINTON STROKE RECOGNITION</div>
          <h1>Turn every rally<br className="mobile-break" /> into <span>insight.</span></h1>
          <p>
            Analyze badminton footage with your trained machine-learning model
            and review its predicted stroke classifications.
          </p>
          <div className="hero-meta">
            <span><Icon name="activity" size={16} /> Five stroke classes</span>
            <span><Icon name="video" size={16} /> Video-based analysis</span>
          </div>
        </section>

        <section className="upload-section">
          <div className="section-heading">
            <div>
              <div className="section-kicker">01 / INPUT</div>
              <h2>{mode === "single" ? "Upload a video" : "Upload video clips"}</h2>
              <p>
                {mode === "single"
                  ? "Choose one badminton clip to identify the stroke."
                  : "Select several clips to analyze them one by one."}
              </p>
            </div>
            <span className="mode-badge">{mode === "single" ? "SINGLE CLIP" : "MULTIPLE CLIPS"}</span>
          </div>

          <input
            ref={inputRef}
            className="hidden-input"
            type="file"
            accept={ACCEPTED_TYPES}
            multiple={mode === "multiple"}
            onChange={(event) => addFiles(event.target.files)}
          />

          {files.length === 0 ? (
            <button
              className={`dropzone ${dragging ? "dragging" : ""}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                addFiles(event.dataTransfer.files);
              }}
            >
              <span className="upload-icon"><Icon name="upload" size={28} /></span>
              <strong>{dragging ? "Drop your video here" : "Choose your footage"}</strong>
              <span className="drop-description">Drag and drop here, or browse your device</span>
              <span className="browse-button">Browse {mode === "single" ? "video" : "videos"}</span>
              <span className="format-list">MP4 <i /> AVI <i /> MOV <i /> MKV</span>
            </button>
          ) : (
            <div className="selected-panel">
              <div className="selected-header">
                <div>
                  <strong>Selected footage</strong>
                  <span>{files.length} {files.length === 1 ? "clip" : "clips"} · {(files.reduce((sum, file) => sum + file.size, 0) / (1024 * 1024)).toFixed(2)} MB</span>
                </div>
                <button className="text-button" onClick={() => inputRef.current?.click()}>
                  Add {mode === "multiple" ? "clips" : "video"}
                </button>
              </div>

              <div className="file-list">
                {files.map((file, index) => (
                  <div className="file-row" key={`${file.name}-${file.lastModified}-${index}`}>
                    <span className="file-icon"><Icon name="file" size={20} /></span>
                    <span className="file-info">
                      <strong>{file.name}</strong>
                      <small>{(file.size / (1024 * 1024)).toFixed(2)} MB</small>
                    </span>
                    <button className="remove-file" onClick={() => removeFile(index)} aria-label={`Remove ${file.name}`}>
                      <Icon name="trash" size={18} />
                    </button>
                  </div>
                ))}
              </div>

              <div className="upload-actions">
                <button className="secondary-button" onClick={resetAnalysis} disabled={loading}>Clear all</button>
                <button className="primary-button" onClick={handlePredict} disabled={loading}>
                  {loading ? "Analyzing..." : `Analyze ${files.length === 1 ? "clip" : `${files.length} clips`}`}
                  {!loading && <Icon name="arrow" size={17} />}
                </button>
              </div>
            </div>
          )}

          {files.length === 0 && error && <div className="error-message"><Icon name="alert" size={18} />{error}</div>}
          {loading && (
            <div className="progress-panel">
              <div className="progress-copy">
                <span>Analyzing videos</span>
                <strong>{progress.done} / {progress.total}</strong>
              </div>
              <div className="progress-track">
                <div style={{ width: `${progress.total ? (progress.done / progress.total) * 100 : 0}%` }} />
              </div>
              <small>Keep the backend running until analysis finishes.</small>
            </div>
          )}
        </section>

        {error && files.length > 0 && <div className="error-message"><Icon name="alert" size={18} />{error}</div>}

        {results.length > 0 && (
          <section className="results-section">
            <div className="section-heading results-heading">
              <div>
                <div className="section-kicker">02 / RESULTS</div>
                <h2>Analysis results</h2>
                <p>Review the predicted stroke and model confidence for each clip.</p>
              </div>
              <button className="secondary-button" onClick={resetAnalysis}>New analysis</button>
            </div>

            <div className="results-list">
              {results.map((result, index) => (
                <article className="result-card" key={`${result.filename}-${index}`}>
                  <div className="result-top">
                    <span className="result-index">{String(index + 1).padStart(2, "0")}</span>
                    <div className="result-file">
                      <strong>{result.filename}</strong>
                      <small>{result.status === "success" ? "Prediction complete" : "Prediction failed"}</small>
                    </div>
                    <span className={`result-status ${result.status}`}>
                      <Icon name={result.status === "success" ? "check" : "alert"} size={18} />
                    </span>
                  </div>

                  {result.status === "success" ? (
                    <>
                      <div className="result-main">
                        <div>
                          <span className="result-label">PREDICTED STROKE</span>
                          <h3>{result.prediction}</h3>
                        </div>
                        <div className="confidence">
                          <span className="result-label">CONFIDENCE</span>
                          <strong>{result.confidence.toFixed(2)}%</strong>
                        </div>
                      </div>
                      {result.probabilities.length > 0 && (
                        <details className="probabilities">
                          <summary>View class probabilities</summary>
                          <div className="probability-list">
                            {result.probabilities.map((item) => (
                              <div className="probability-row" key={item.label}>
                                <span>{item.label}</span>
                                <div className="probability-track">
                                  <div style={{ width: `${Math.max(0, Math.min(100, item.value))}%` }} />
                                </div>
                                <strong>{item.value.toFixed(2)}%</strong>
                              </div>
                            ))}
                          </div>
                        </details>
                      )}
                    </>
                  ) : (
                    <div className="result-error">{result.message}</div>
                  )}
                </article>
              ))}
            </div>
          </section>
        )}

        <footer className="footer">
          <span><span className="status-dot" /> RallyAI local workspace</span>
          <span>Sports intelligence · Stroke recognition</span>
        </footer>
      </main>

      {dialog && (
        <div className="dialog-backdrop" onMouseDown={(event) => {
          if (event.target === event.currentTarget) setDialog("");
        }}>
          <section className="dialog-card" role="dialog" aria-modal="true" aria-labelledby="dialog-title">
            <div className="dialog-header">
              <div className="dialog-title-icon">
                <Icon name={dialog === "about" ? "info" : dialog === "help" ? "help" : "trophy"} size={23} />
              </div>
              <button className="dialog-close" onClick={() => setDialog("")} aria-label="Close dialog">
                <Icon name="close" />
              </button>
            </div>

            {dialog === "about" && (
              <>
                <div className="section-kicker">PROJECT OVERVIEW</div>
                <h2 id="dialog-title">About RallyAI</h2>
                <p className="dialog-intro">
                  RallyAI is a badminton stroke-recognition project designed to
                  analyze video clips and help review stroke classifications.
                </p>
                <div className="about-grid">
                  <div><span>01</span><strong>Video input</strong><p>Accepts supported badminton video clips.</p></div>
                  <div><span>02</span><strong>Frame processing</strong><p>OpenCV prepares frames for model inference.</p></div>
                  <div><span>03</span><strong>Stroke recognition</strong><p>The trained CNN-LSTM model predicts a stroke class.</p></div>
                  <div><span>04</span><strong>Results</strong><p>Displays the prediction and probabilities returned by the backend.</p></div>
                </div>
                <div className="class-chips">
                  {CLASS_NAMES.map((name) => <span key={name}>{name.replace("_", " ")}</span>)}
                </div>
                <p className="dialog-note">
                  Technologies: React, Python, FastAPI, PyTorch and OpenCV.
                  Model predictions are estimates and depend on video quality and training data.
                </p>
              </>
            )}

            {dialog === "help" && (
              <>
                <div className="section-kicker">QUICK GUIDE</div>
                <h2 id="dialog-title">How to use RallyAI</h2>
                <p className="dialog-intro">Follow these steps to analyze your badminton clips.</p>
                <div className="help-steps">
                  <div><span>1</span><div><strong>Choose a mode</strong><p>Use the settings gear to select Single clip or Multiple clips. Single clip is the default.</p></div></div>
                  <div><span>2</span><div><strong>Select your videos</strong><p>Browse or drag supported MP4, AVI, MOV or MKV files into the upload area.</p></div></div>
                  <div><span>3</span><div><strong>Start analysis</strong><p>Click Analyze. The website sends each selected clip to the local prediction backend.</p></div></div>
                  <div><span>4</span><div><strong>Review results</strong><p>Check the predicted stroke, confidence score and class probabilities when provided.</p></div></div>
                </div>
                <div className="help-tip">
                  <Icon name="info" size={19} />
                  <p>Keep the FastAPI backend running at <strong>127.0.0.1:8000</strong>. If a request fails, check the backend terminal and confirm that its /predict endpoint accepts the field named <code>file</code>.</p>
                </div>
                <p className="dialog-note">Complete match analysis is not available yet.</p>
              </>
            )}

            {dialog === "coming-soon" && (
              <>
                <div className="coming-soon-icon"><Icon name="trophy" size={28} /></div>
                <div className="section-kicker">FEATURE NOT AVAILABLE</div>
                <h2 id="dialog-title">Complete match analysis</h2>
                <p className="dialog-intro">
                  This feature is not enabled yet. The Spark team has not released
                  complete-match analysis.
                </p>
                <button className="primary-button dialog-action" onClick={() => setDialog("")}>Understood</button>
              </>
            )}
          </section>
        </div>
      )}
    </div>
  );
}

export default App;
