import fs from 'fs';
import path from 'path';

const distDir = 'dist';
const oldPath = path.join(distDir, 'Index.html');
const newPath = path.join(distDir, 'index.html');

try {
  if (fs.existsSync(oldPath)) {
    fs.renameSync(oldPath, newPath);
    console.log('Renamed Index.html to index.html');
  }
} catch (err) {
  console.error('Error renaming file:', err.message);
}
