const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const output = document.getElementById('emotion-output');

const API_URL = 'http://localhost:8000/predict';

navigator.mediaDevices.getUserMedia({ video: true })
  .then((stream) => { video.srcObject = stream; })
  .catch((err) => { output.textContent = 'Camera access denied: ' + err.message; });

function captureFrame() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.8);
}

async function sendFrame() {
  if (!video.videoWidth) return; // camera not ready yet
  const dataUrl = captureFrame();

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl }),
    });
    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    const data = await response.json();
    renderPredictions(data.faces);
  } catch (err) {
    output.textContent = 'Prediction error: ' + err.message;
  }
}

function renderPredictions(faces) {
  if (!faces || faces.length === 0) {
    output.textContent = 'No face detected';
    return;
  }
  faces.forEach((face) => {
    const { x, y, w, h } = face.box;
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = '#00ff88';
    ctx.font = '16px sans-serif';
    ctx.fillText(`${face.label} (${(face.confidence * 100).toFixed(0)}%)`, x, y - 8);
  });
  canvas.style.display = 'block';
  output.textContent = faces
    .map((f) => `${f.label} — ${(f.confidence * 100).toFixed(1)}%`)
    .join(' | ');
}

setInterval(sendFrame, 1000);
