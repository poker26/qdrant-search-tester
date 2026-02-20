const path = require('path');
const fs = require('fs');

const rootDir = __dirname;
const venvPython = path.join(rootDir, 'venv/bin/python');
const venvPythonParent = path.join(rootDir, '..', 'venv/bin/python');
const pythonPath = fs.existsSync(venvPython) ? venvPython : venvPythonParent;
const cwd = fs.existsSync(venvPython) ? rootDir : path.join(rootDir, '..');

module.exports = {
  apps: [
    {
      name: 'qdrant-tester',
      script: pythonPath,
      args: '-m streamlit run streamlit_dashboard/test_dashboard.py --server.port 8501 --server.address 0.0.0.0',
      cwd: cwd,
      interpreter: 'none',
      env: {
        NODE_ENV: 'production'
      }
    }
  ]
};
