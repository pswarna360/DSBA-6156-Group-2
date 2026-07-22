# Education ROI Streamlit Supplement

This Streamlit app is a professional presentation supplement for the Education ROI Intelligence project. It combines the model, data pipeline, EDA, results, and pathway persona analysis in one interactive deliverable.

## Run

```bash
# From the folder containing this README:
./launch_streamlit.sh
```

The launcher uses the local `.venv`, enforces the pinned requirements, and automatically chooses the first open Streamlit port starting at `8501`. To force a specific port:

```bash
STREAMLIT_PORT=8503 ./launch_streamlit.sh
```

The app reads precomputed prediction and persona fields from `streamlit_assets/presentation_data.parquet`, so no separate model server or secrets are required.

## Deploy to Streamlit Community Cloud

Push this repository to GitHub, then create a new app at <https://share.streamlit.io>.

- Repository: `pswarna360/DSBA-6156-Group-2`
- Branch: `main`
- Main file path: `streamlit_app.py`
- Python version: use the Streamlit Cloud default, or select `3.14` in Advanced settings

The app avoids live scikit-learn/joblib loading on Streamlit Cloud. Prediction and persona fields are precomputed in `streamlit_assets/presentation_data.parquet`, and the scenario lab uses a lightweight NumPy estimator for interactive what-if inputs.

No Streamlit secrets are required for this app. The deployed app uses the static files in `streamlit_assets/`.

The app expects these files in `streamlit_assets/`:

- `presentation_data.parquet`

## Application Sections

Use the sidebar to move through the analysis sections:

1. Overview
2. Data Foundation
3. Exploration
4. Model Lab
5. Performance
6. Personas
7. Decision Takeaways

The sidebar filters support interactive exploration by job zone, credential level, and salary range.
