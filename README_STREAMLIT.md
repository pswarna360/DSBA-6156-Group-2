# Education ROI Streamlit Supplement

This Streamlit app is a professional presentation supplement for the Education ROI Intelligence project. It combines the model, data pipeline, EDA, results, and pathway persona analysis in one interactive deliverable.

## Run

```bash
./launch_streamlit.sh
```

The launcher uses the local `.venv`, installs the pinned requirements if needed, and automatically chooses the first open Streamlit port starting at `8501`. To force a specific port:

```bash
STREAMLIT_PORT=8503 ./launch_streamlit.sh
```

## Deploy to Streamlit Community Cloud

Push this repository to GitHub, then create a new app at <https://share.streamlit.io>.

- Repository: `pswarna360/DSBA-6156-Group-2`
- Branch: `main`
- Main file path: `streamlit_app.py`
- Python version: select `3.10` in Advanced settings

No Streamlit secrets are required for this app. The deployed app uses the static files in `streamlit_assets/`.

The app expects these files in `streamlit_assets/`:

- `presentation_data.parquet`
- `optimized_hgbr_model.joblib`
- `kmeans_personas.joblib`
- `kmeans_scaler.joblib`

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
