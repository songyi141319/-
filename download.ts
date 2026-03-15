import fs from 'fs';
import https from 'https';
import path from 'path';

const dir = path.join(process.cwd(), 'public', 'models');
fs.mkdirSync(dir, { recursive: true });

const file = fs.createWriteStream(path.join(dir, 'pose_landmarker_lite.task'));
https.get('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task', (response) => {
  if (response.statusCode === 302 || response.statusCode === 301) {
    https.get(response.headers.location!, (res) => {
      res.pipe(file);
      file.on('finish', () => {
        file.close();
        console.log('Download complete');
      });
    });
  } else {
    response.pipe(file);
    file.on('finish', () => {
      file.close();
      console.log('Download complete');
    });
  }
}).on('error', (err) => {
  console.error('Error downloading:', err.message);
});
