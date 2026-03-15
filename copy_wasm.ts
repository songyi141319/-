import fs from 'fs';
import path from 'path';

const srcDir = path.join(process.cwd(), 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const destDir = path.join(process.cwd(), 'public', 'wasm');

fs.mkdirSync(destDir, { recursive: true });

fs.readdirSync(srcDir).forEach(file => {
  fs.copyFileSync(path.join(srcDir, file), path.join(destDir, file));
  console.log(`Copied ${file}`);
});
