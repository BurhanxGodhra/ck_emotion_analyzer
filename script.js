const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const emotionDisplay = document.getElementById('emotion');

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(error => {
        console.error('Error accessing camera:', error);
    });

function captureImage() {
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Send the image to the backend for prediction
    const imageData = canvas.toDataURL('image/jpeg');
    fetch('/predict', {
        method: 'POST',
        body: JSON.stringify({ image: imageData })
    })
    .then(response => response.json())
    .then(data => {
        emotionDisplay.textContent = `Predicted Emotion: ${data.emotion}`;
    })
    .catch(error => {
        console.error('Error sending image:', error);
    });
}

setInterval(captureImage, 1000);