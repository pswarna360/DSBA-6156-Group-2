# Education ROI Streamlit Supplement

This Streamlit app is a professional presentation supplement for the Education ROI Intelligence project. It combines the model, data pipeline, EDA, results, and pathway persona analysis in one interactive deliverable.

## Run

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

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
