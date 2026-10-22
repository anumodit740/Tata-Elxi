# Predictive Maintenance — AI-FBI

NASA C-MAPSS engine-health monitoring for Tata Elxsi Teliport, Case 1.

**Team:** Anumodit Shukla · Yogesh Singh · Anoop Kumar

The Streamlit app estimates remaining useful life (RUL), displays abnormal-behaviour signals, and creates maintenance work orders using a rule-based agent. It runs the retained models for FD001–FD004; it does not train models.

The interface combines Streamlit input controls with a locally bundled **React, JavaScript and CSS** dashboard. It includes interactive RUL charts, maintenance histories, signal diagnostics and downloadable evidence. All chart points come from the current inference result; no sample chart values are generated in the frontend.

## Run locally

Use **Python 3.12**. From the extracted project folder:

```bash
python -m pip install -r requirements.txt
python -B run.py
```

Open **http://127.0.0.1:8501**. If the port is busy:

```bash
python -B run.py --port 8502
```

The launcher checks the artifact hashes before starting Streamlit. You can run that check separately:

```bash
python -B verify_package.py
```

## Use the dashboard

1. Select the subset that matches your input.
2. Choose **NASA TRAIN example** or upload a trajectory.
3. Select an engine and the last observed cycle, then press **Run model and agent**.
4. Inspect the RUL timeline and the **Work orders** tab.
5. Press **Retry identical prefix** to check that the same observations do not create duplicate orders.
6. Download the work orders, predictions or inference receipt before ending the session.

Use **Start new session** before moving back to an earlier cycle. Each session keeps its own temporary agent state.

### Input format

- Canonical-header CSV, or whitespace-separated NASA trajectory text with 26 numeric columns: engine ID, cycle, three operating settings and 21 sensors.
- Supply the complete available history for each engine, starting at cycle 1, in consecutive cycle order.
- Missing/non-finite values, duplicate keys, gaps, reversed histories and incorrect schemas are rejected.
- Demo uploads are limited to 5 MB and 5,000 observations. The app processes one selected engine at a time.

Raw data is not included. The example loader downloads the recorded NASA archive, checks its SHA-256 and opens only the requested TRAIN member. Initial example loading needs internet access; upload a compatible trajectory if the download is unavailable.

## Model and maintenance logic

```text
Observed history → validation → causal features → frozen model
                                           ↓
                          RUL → rule agent → persisted work order
                                           ↓
                                  Streamlit dashboard
```

- **RUL:** retained HistGradientBoosting models with a 115-cycle target cap. Features use history through the observed cycle.
- **Abnormal behaviour:** robust MAD deviation and a classifier for the RUL ≤30 proxy. Both are informational; they do not create orders in the retained policy.
- **Maintenance:** RUL thresholds of 30 cycles for CRITICAL and 60 for WARNING, two-cycle confirmation and 10-cycle hysteresis. An emergency rule applies at ≤5 cycles. Orders remain open until human closure; this app does not provide a closure control.
- **Integrity:** model hashes are checked before loading. Identical-event retries reuse the existing state and order.
- **No LLM:** predictions and actions use the packaged models and explicit rules.

## Project files

| File/folder | Purpose |
|---|---|
| `app.py` | Streamlit controls and session orchestration |
| `ui.py` | Python bridge to the React component |
| `frontend/` | React source, compiled JavaScript, CSS and third-party licenses |
| `service.py` | Input validation, inference and agent integration |
| `agent_v3.py` | Maintenance rules and persisted state |
| `model/` | Four retained models, runtime code and model manifest |
| `examples/provenance.json` | TRAIN example source and expected hashes |
| `run.py` | Integrity check and local launch |
| `verify_package.py`, `ARTIFACTS.json` | Package hash verification |
| `requirements.txt` | Pinned dependencies |

For a hosted Streamlit app, the entry point is `app.py`. Upload the complete project, including `ui.py` and `frontend/`; replacing `app.py` alone is not sufficient. Keep `requirements.txt`, `model/` and `examples/` at the repository root, and use Python 3.12. The compiled frontend is included, so deployment needs no Node.js server, npm installation or external CDN.

### Optional frontend development

To modify and rebuild the React source, use Node.js and pnpm from the `frontend/` folder:

```bash
pnpm install --frozen-lockfile
pnpm run build
```

Restart Streamlit after changing the frontend assets. Only frontend development needs these commands. Keep `frontend/THIRD_PARTY_LICENSES.txt` with the bundled assets.

## Verification and limits

Local checks covered all four models, causal-prefix parity, real agent integration, identical retries, invalid inputs and Streamlit interaction tests. The included examples are fitting-TRAIN histories: they demonstrate runtime behaviour, **not held-out accuracy**.

All official TEST sets were previously inspected. Historical TEST results remain **benchmark, previously inspected; no selection use**. No test metrics or new accuracy claims are made here. The near-failure classifier uses an RUL proxy, not physical fault-onset labels.

Public hosting and a clean cloud installation remain unverified. Session storage is temporary; download your outputs before a restart. Broad ingestion-review integration and rollback remain planned. The full retraining/research workflow is outside this minimal inference package.

## License

Project code is provided under the [MIT License](LICENSE). NASA data is downloaded separately and is not bundled with this repository.
