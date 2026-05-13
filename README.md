# Spin Glass / Ising Market Model

Statistical physics model of financial markets.  
Each ETF = binary spin (+1 / -1) based on sign of daily return.  
Interaction matrix = correlation of returns (rolling 60 days).  
Local field \( h_i = \sum_j J_{ij} s_j \) measures alignment with the current metastable state.  
High positive local field → overweight. Magnetisation = market consensus.

- **Output:** Top 3 ETFs by local field per universe
- **Diagnostics:** Magnetisation, energy, effective temperature
- Runs daily on GitHub Actions

## Local execution
```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
