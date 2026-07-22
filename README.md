\# DARE Reproduction \& Extension



Reproduction of "Evaluating the reliability of machine-learning-based predictions used in nuclear power plant instrumentation and control systems" (Chen, Bao, Dinh — Reliability Engineering \& System Safety, 2024), with a proposed improvement to reduce over rejection via adaptive bandwidth.



\# Status

Work in progress. Phase 0 (setup) complete.



\# Structure

\- `src/` — DARE implementation, metrics, digital twin regressor

\- `tests/` — sanity checks on core math

\- `notebooks/` — validation against paper's Table 5, result plots

\- `data/` — synthetic data from paper's VAE regenerator (gitignored)



\# Setup

python -m venv .venv

.venv\\Scripts\\Activate.ps1 # Windows

pip install -r requirements.txt

