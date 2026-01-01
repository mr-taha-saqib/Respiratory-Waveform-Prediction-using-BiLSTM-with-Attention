"""
Respiratory Waveform Prediction - 15 Second NPRE Prediction
============================================================

This script predicts 15 seconds of NPRE (nasal pressure) waveform from an EDF file.

Performance: 50.1% mean correlation / 53.2% median correlation

Usage:
    python predict_15s.py input.EDF output_basename

Output:
    - output_basename.csv (for inspection/debugging)
    - output_basename.EDF (for event detection systems)

Requirements:
    - numpy
    - pandas
    - scipy
    - pyedflib
    - tensorflow (2.x)
"""

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import pandas as pd
import pickle
from scipy import signal
from scipy.fft import fft, fftfreq
import pyedflib
import tensorflow as tf
from tensorflow.keras import models, layers


# GPU/CPU Device Configuration
def configure_device():
    """
    Configure TensorFlow to use GPU if available, otherwise fall back to CPU.
    Returns device information for logging.
    """
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Enable memory growth to prevent GPU memory allocation issues
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            device_info = {
                'device_type': 'GPU',
                'device_count': len(gpus),
                'device_names': [gpu.name for gpu in gpus],
                'estimated_speedup': '10-50x faster than CPU'
            }
        except RuntimeError as e:
            # GPU configuration failed, fall back to CPU
            device_info = {
                'device_type': 'CPU',
                'device_count': 1,
                'device_names': ['CPU'],
                'estimated_speedup': 'baseline',
                'gpu_error': str(e)
            }
    else:
        device_info = {
            'device_type': 'CPU',
            'device_count': 1,
            'device_names': ['CPU'],
            'estimated_speedup': 'baseline',
            'note': 'No GPU detected. Install CUDA/cuDNN for NVIDIA GPU support.'
        }

    return device_info


# Custom Layer Definition
class BahdanauAttention(layers.Layer):
    """
    Bahdanau Attention mechanism for sequence modeling.
    Allows the model to focus on relevant parts of the input sequence.
    Supports mixed precision training (float16/float32).
    """
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def build(self, input_shape):
        # Build the sublayers with correct input shapes
        # input_shape: (batch_size, timesteps, features)
        features_dim = input_shape[-1]

        # W1 processes the full sequence
        self.W1.build((None, None, features_dim))
        # W2 processes the last hidden state
        self.W2.build((None, 1, features_dim))
        # V processes the combined representation
        self.V.build((None, None, self.units))

        super().build(input_shape)

    def call(self, features):
        # Cast to float32 for computation, then cast back to input dtype
        input_dtype = features.dtype
        if input_dtype == tf.float16:
            features = tf.cast(features, tf.float32)

        hidden_with_time = tf.expand_dims(features[:, -1, :], 1)
        score = self.V(tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time)
        ))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # Cast back to input dtype if necessary
        if input_dtype == tf.float16:
            context_vector = tf.cast(context_vector, tf.float16)

        return context_vector

    def compute_output_shape(self, input_shape):
        # Output shape: (batch_size, input_features)
        return (input_shape[0], input_shape[2])

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


# Configuration
TARGET_FS = 32
INPUT_SAMPLES = 640  # 20 seconds
OUTPUT_SAMPLES = 160  # 5 seconds
STRIDE = 80
NUM_CHANNELS = 5

NASAL_NAMES = ["NPRE", "Druck Flow", "CPAP Flow"]
AIRFLOW_NAMES = ["Flow Th", "Therm"]
ABD_NAMES = ["ABD", "Abdomen", "RIP.Abdom"]
THX_NAMES = ["Thorax", "RIP.Thrx"]


def find_channel(signal_labels, possible_names):
    """Find channel index from list of possible names."""
    for name in possible_names:
        if name in signal_labels:
            return signal_labels.index(name)
    raise ValueError(f"Channel not found. Looking for {possible_names}, found: {signal_labels}")


def lowpass_filter(data, fs, cutoff=4.0, numtaps=513):
    """Apply FIR lowpass filter."""
    nyq = fs / 2.0
    taps = signal.firwin(numtaps, cutoff / nyq, window='hamming')
    return signal.filtfilt(taps, 1.0, data)


def highpass_filter(data, fs, cutoff=0.005):
    """Apply highpass filter for drift removal."""
    sos = signal.butter(3, cutoff, btype='high', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, data)


def extract_breathing_frequency(data, fs=32, window_samples=640):
    """Extract instantaneous breathing frequency from NPRE signal."""
    breathing_freq = np.zeros(len(data))
    hop = window_samples // 4

    for start in range(0, len(data) - window_samples + 1, hop):
        end = start + window_samples
        window = data[start:end]

        fft_vals = fft(window)
        freqs = fftfreq(len(window), 1/fs)

        mask = (freqs >= 0.1) & (freqs <= 0.7)
        breathing_range_fft = np.abs(fft_vals[mask])
        breathing_range_freqs = freqs[mask]

        if len(breathing_range_fft) > 0:
            dom_idx = np.argmax(breathing_range_fft)
            dom_freq = breathing_range_freqs[dom_idx]
            breathing_freq[start:end] = dom_freq

    if breathing_freq[-1] == 0:
        breathing_freq[breathing_freq == 0] = np.mean(breathing_freq[breathing_freq > 0])

    return breathing_freq


def write_edf_output(predictions, output_file, fs=32):
    """
    Write predictions to EDF file format for event detection systems.

    Args:
        predictions: List of 15-second prediction arrays
        output_file: Output EDF file path
        fs: Sampling frequency (default 32 Hz)
    """
    # Concatenate all predictions into continuous signal
    continuous_signal = np.concatenate(predictions)

    # Create EDF file
    try:
        f = pyedflib.EdfWriter(output_file, 1, file_type=pyedflib.FILETYPE_EDFPLUS)

        # Configure signal header
        channel_info = {
            'label': 'NPRE_predicted',
            'dimension': 'mV',
            'sample_frequency': fs,
            'physical_max': float(np.max(continuous_signal)),
            'physical_min': float(np.min(continuous_signal)),
            'digital_max': 32767,
            'digital_min': -32768,
            'transducer': 'AI Model V1.6',
            'prefilter': 'BiLSTM+Attention'
        }

        f.setSignalHeader(0, channel_info)
        f.writeSamples([continuous_signal])
        f.close()

        return True
    except Exception as e:
        print(f"Warning: Could not write EDF file: {e}")
        return False


def process_edf(filepath):
    """Process EDF file and extract 5 channels."""
    print(f"Processing: {filepath}")

    with pyedflib.EdfReader(filepath) as f:
        signal_labels = f.getSignalLabels()

        nasal_idx = find_channel(signal_labels, NASAL_NAMES)
        airflow_idx = find_channel(signal_labels, AIRFLOW_NAMES)
        abd_idx = find_channel(signal_labels, ABD_NAMES)
        thx_idx = find_channel(signal_labels, THX_NAMES)

        results = {}

        for idx, output_name in [(nasal_idx, 'NPRE'), (airflow_idx, 'Flow Th'),
                                  (abd_idx, 'ABD'), (thx_idx, 'Thorax')]:
            data = f.readSignal(idx)
            fs = f.getSampleFrequency(idx)

            data = data - np.median(data)
            data = lowpass_filter(data, fs, cutoff=4.0)

            if fs != TARGET_FS:
                data = signal.resample_poly(data, TARGET_FS, int(fs))

            if output_name in ['ABD', 'Thorax']:
                data = highpass_filter(data, TARGET_FS, cutoff=0.005)

            results[output_name] = data

        breathing_freq = extract_breathing_frequency(results['NPRE'], fs=TARGET_FS, window_samples=INPUT_SAMPLES)
        results['Breathing_Freq'] = breathing_freq

        min_len = min(len(v) for v in results.values())
        for k in results:
            results[k] = results[k][:min_len]

        df = pd.DataFrame(results)
        df = df[['Flow Th', 'NPRE', 'ABD', 'Thorax', 'Breathing_Freq']]

    print(f"Extracted {len(df)} samples ({len(df)/TARGET_FS:.1f} seconds)")
    return df


def predict_15s(model, X_data, start_idx):
    """
    Predict 15 seconds of NPRE waveform.

    Uses three separate 20s input windows to predict three consecutive
    5s segments, which are concatenated to form the 15s output.
    """
    predictions = []
    indices_per_5s = OUTPUT_SAMPLES // STRIDE

    for step in range(3):
        window_idx = start_idx + (step * indices_per_5s)

        if window_idx < len(X_data):
            window = X_data[window_idx]
            pred = model.predict(window.reshape(1, INPUT_SAMPLES, NUM_CHANNELS), verbose=0)
            predictions.append(pred.flatten())
        else:
            if predictions:
                predictions.append(predictions[-1])
            else:
                predictions.append(np.zeros(OUTPUT_SAMPLES))

    return np.concatenate(predictions)


def main():
    if len(sys.argv) != 3:
        print("Usage: python predict_15s.py <input.EDF> <output_basename>")
        print("\nOutputs:")
        print("  - output_basename.csv (for inspection)")
        print("  - output_basename.EDF (for event detection)")
        print("\nExample:")
        print("  python predict_15s.py patient_data.EDF predicted_waveform")
        print("  Creates: predicted_waveform.csv and predicted_waveform.EDF")
        sys.exit(1)

    input_file = sys.argv[1]
    output_basename = sys.argv[2]

    # Generate output filenames
    output_csv = output_basename if output_basename.endswith('.csv') else f"{output_basename}.csv"
    output_edf = output_basename.replace('.csv', '.EDF') if output_basename.endswith('.csv') else f"{output_basename}.EDF"

    print("="*70)
    print("    RESPIRATORY WAVEFORM PREDICTION - 15 SECOND NPRE")
    print("="*70)

    # Configure GPU/CPU device
    device_info = configure_device()
    print(f"\nCompute Device: {device_info['device_type']}")
    if device_info['device_type'] == 'GPU':
        print(f"  GPU Count: {device_info['device_count']}")
        print(f"  Estimated Performance: {device_info['estimated_speedup']}")
    else:
        print(f"  {device_info.get('note', 'Using CPU for inference')}")
        if 'gpu_error' in device_info:
            print(f"  GPU Error: {device_info['gpu_error']}")

    # Load model
    print("\nLoading model...")
    try:
        model = models.load_model(
            'model/best_model_v1.6_optimized_with_15s.keras',
            custom_objects={'BahdanauAttention': BahdanauAttention},
            compile=False
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\n" + "="*70)
        print("TROUBLESHOOTING:")
        print("="*70)
        print("This error usually means you're running the script from the wrong directory.")
        print("\nCORRECT WAY:")
        print("  1. Open Command Prompt (cmd.exe)")
        print("  2. Navigate TO the CLIENT_DELIVERY folder:")
        print("     cd path\\to\\CLIENT_DELIVERY")
        print("  3. Run: python predict_15s.py input.EDF output")
        print("\nTo verify you're in the correct directory:")
        print("  Run: dir")
        print("  You should see: predict_15s.py, model (folder), README.md")
        print("\nSee HOW_TO_RUN.txt for more details.")
        print("="*70)
        sys.exit(1)

    # Load normalization stats
    print("Loading normalization parameters...")
    try:
        with open('model/normalization_stats_v1.6.pkl', 'rb') as f:
            norm_stats = pickle.load(f)

        # Handle both old and new pickle formats
        if 'means' in norm_stats:
            means = pd.Series(norm_stats['means'])
            stds = pd.Series(norm_stats['stds'])
        elif 'channel_means' in norm_stats:
            # Convert arrays to Series with proper channel names
            channel_names = norm_stats.get('channel_names', ['Flow_Th', 'NPRE', 'ABD', 'Thorax', 'Breathing_Frequency'])
            # Map channel names to match dataframe columns
            channel_map = {
                'Flow_Th': 'Flow Th',
                'NPRE': 'NPRE',
                'ABD': 'ABD',
                'Thorax': 'Thorax',
                'Breathing_Frequency': 'Breathing_Freq'
            }
            mapped_names = [channel_map.get(name, name) for name in channel_names]
            means = pd.Series(norm_stats['channel_means'], index=mapped_names)
            stds = pd.Series(norm_stats['channel_stds'], index=mapped_names)
        else:
            raise KeyError("Normalization stats file missing required keys")

        print("Normalization parameters loaded")
    except Exception as e:
        print(f"Error loading normalization stats: {e}")
        sys.exit(1)

    # Process EDF
    print("\n" + "="*70)
    df = process_edf(input_file)

    # Normalize
    print("\nApplying normalization...")
    df_normalized = (df - means) / stds

    # Create windows
    print("Creating prediction windows...")
    input_channels = ['Flow Th', 'NPRE', 'ABD', 'Thorax', 'Breathing_Freq']

    # Extract to numpy array once to avoid repeated DataFrame indexing
    data_array = df_normalized[input_channels].values

    X_windows = []
    for start in range(0, len(data_array) - INPUT_SAMPLES, STRIDE):
        end = start + INPUT_SAMPLES
        window = data_array[start:end]
        if len(window) == INPUT_SAMPLES:
            X_windows.append(window)

    X_data = np.array(X_windows, dtype=np.float32)
    print(f"Created {len(X_data)} windows")

    # Make predictions
    print("\nGenerating 15-second predictions...")
    indices_per_5s = OUTPUT_SAMPLES // STRIDE
    skip_indices = indices_per_5s * 3

    predictions_15s = []
    time_starts = []

    for i in range(0, len(X_data) - skip_indices - 1, skip_indices):
        pred_15s = predict_15s(model, X_data, i)
        predictions_15s.append(pred_15s)
        time_starts.append(i * STRIDE / TARGET_FS)

    print(f"Generated {len(predictions_15s)} 15-second predictions")

    if len(predictions_15s) == 0:
        print("\nError: Input EDF file too short for 15-second predictions")
        print("Minimum required: ~35 seconds of data")
        sys.exit(1)

    # Denormalize
    print("\nDenormalizing predictions...")
    npre_mean = means['NPRE']
    npre_std = stds['NPRE']

    predictions_denorm = [(pred * npre_std) + npre_mean for pred in predictions_15s]

    # Save CSV results (for inspection)
    print(f"\nSaving CSV: {output_csv}")
    results = []
    for i, (pred, start_time) in enumerate(zip(predictions_denorm, time_starts)):
        for j, value in enumerate(pred):
            time = start_time + (j / TARGET_FS)
            results.append({
                'prediction_index': i,
                'time_seconds': time,
                'npre_predicted': value
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"CSV saved: {len(results_df):,} samples")

    # Save EDF results (for event detection)
    print(f"\nSaving EDF: {output_edf}")
    edf_success = write_edf_output(predictions_denorm, output_edf, fs=TARGET_FS)
    if edf_success:
        print(f"EDF saved: {len(predictions_denorm)} predictions ({len(predictions_denorm) * 15}s continuous)")
    else:
        print("EDF output failed (CSV still available)")

    print("\n" + "="*70)
    print("    PREDICTION COMPLETE")
    print("="*70)
    print(f"\nInput:  {input_file}")
    print(f"Output: {output_csv}")
    print(f"        {output_edf}")
    print(f"\nTotal predictions: {len(predictions_15s)} × 15s")
    print(f"Total time covered: {len(predictions_15s) * 15} seconds")
    print(f"Total samples: {len(results_df):,}")
    print("\nExpected accuracy: 50.1% correlation (validated)")
    print("\nFormat usage:")
    print("  - CSV: For inspection/debugging")
    print("  - EDF: For event detection systems")
    print("="*70)


if __name__ == "__main__":
    main()
