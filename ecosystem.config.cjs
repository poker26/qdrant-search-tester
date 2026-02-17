const path = require('path');

module.exports = {
  apps: [
    {
      name: 'qdrant-tester',
      script: path.join(__dirname, 'venv/bin/python'),
      args: '-m streamlit run streamlit_dashboard/test-dashboard.py --server.port 8501 --server.address 0.0.0.0',
      cwd: __dirname,
      interpreter: 'none',
      env: {
        NODE_ENV: 'production'
      }
    }
  ]
};
