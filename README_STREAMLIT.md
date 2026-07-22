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

If you see a scikit-learn/joblib artifact error such as `_RemainderColsList`, the app is probably being launched with a different Python environment, such as Anaconda. Use `./launch_streamlit.sh` from this folder so the model loads with the pinned scikit-learn version in `requirements.txt`.

## Deploy to Streamlit Community Cloud

Push this repository to GitHub, then create a new app at <https://share.streamlit.io>.

- Repository: `pswarna360/DSBA-6156-Group-2`
- Branch: `main`
- Main file path: `streamlit_app.py`
- Python version: use the Streamlit Cloud default, or select `3.14` in Advanced settings

The bundled model artifacts are saved with the scikit-learn version pinned in `requirements.txt`. If you change `scikit-learn`, resave the files in `streamlit_assets/` with the same version before deploying.

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
