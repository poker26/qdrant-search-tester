const path = require('path');
const fs = require('fs');

const rootDir = __dirname;
const nestedDir = path.join(rootDir, 'qdrant-search-tester');

// Ищем venv и streamlit_dashboard в разных возможных расположениях
const candidates = [
  { cwd: rootDir, venv: path.join(rootDir, 'venv/bin/python') },
  { cwd: nestedDir, venv: path.join(nestedDir, 'venv/bin/python') },
  { cwd: path.join(rootDir, '..'), venv: path.join(rootDir, '..', 'venv/bin/python') }
];

const dashboardPath = 'streamlit_dashboard/test_dashboard.py';
let pythonPath = null;
let cwd = rootDir;

for (const { cwd: dir, venv } of candidates) {
  const scriptPath = path.join(dir, dashboardPath);
  if (fs.existsSync(venv) && fs.existsSync(scriptPath)) {
    pythonPath = venv;
    cwd = dir;
    break;
  }
}

if (!pythonPath) {
  // Fallback: используем первый найденный venv
  for (const { cwd: dir, venv } of candidates) {
    if (fs.existsSync(venv)) {
      pythonPath = venv;
      cwd = fs.existsSync(path.join(dir, dashboardPath)) ? dir : rootDir;
      break;
    }
  }
}

if (!pythonPath) {
  pythonPath = path.join(rootDir, 'venv/bin/python');
}

module.exports = {
  apps: [
    {
      name: 'qdrant-tester',
      script: pythonPath,
      args: '-m streamlit run streamlit_dashboard/test_dashboard.py --server.port 8501 --server.address 0.0.0.0',
      cwd: cwd,
      interpreter: 'none',
      env: {
        NODE_ENV: 'production',
        STREAMLIT_SERVER_BASE_URL_PATH: '/qtester'
      }
    }
  ]
};
