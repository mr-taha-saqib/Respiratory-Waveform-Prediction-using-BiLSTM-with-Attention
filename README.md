# Respiratory Waveform Prediction (15s NPRE Forecasting)

This project predicts **15 seconds of nasal pressure (NPRE) respiratory waveform**
from physiological signals stored in **EDF files**.

A **BiLSTM model with Bahdanau Attention** is used for time-series forecasting
of respiratory patterns to support downstream clinical analysis.

The pipeline includes **signal filtering, normalization, resampling (32 Hz)**,
breathing frequency extraction, and sliding-window inference.

The model takes **20 seconds of input data** and predicts **15 seconds of future waveform**.
Predictions are exported in both **CSV** and **EDF** formats.

The script supports **GPU acceleration with CPU fallback** and is designed for
efficient inference on real-world biomedical data.

**Author:** Taha Saqib
