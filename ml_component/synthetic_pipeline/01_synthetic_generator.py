"""
Standalone Synthetic ECG Generator for High-Grade AV Block
==========================================================
Generates a 1,500-record dataset of purely synthetic 12-lead ECG signals.
Specifically models the hardest failure cases of the LUDB dataset:
  1. 3rd-Degree AV Block: P-waves are completely dissociated from QRS complexes.
  2. Overlapping Waves: P-waves physically buried inside T-waves or QRS complexes.
  3. Strict isolation: This dataset is saved away from LUDB and loaded independently.

Output:
    - signals.npy       (Shape: 1500, 12, 5000)
    - annotations.pkl   (List of dicts matching LUDB format)
"""

import os
import numpy as np
import pickle
import math

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_RECORDS = 1500
FS = 500
DURATION = 10
N_SAMPLES = FS * DURATION
N_LEADS = 12

def generate_synthetic_record():
    """Generates a 12-lead 10-second ECG simulating dissociated P-waves."""
    signal = np.zeros((N_LEADS, N_SAMPLES))
    annotations = {'p_waves': []}
    
    # ── 1. Simulate Ventricular Rhythm (QRS & T) ─────────────────────────────
    # Ventricular rate in complete heart block is slow (e.g., 40 bpm)
    v_rate = np.random.uniform(35, 50) 
    v_interval = int((60 / v_rate) * FS)
    v_positions = np.arange(v_interval, N_SAMPLES - v_interval, v_interval)
    
    for pos in v_positions:
        # QRS complex (simple spike for simulation)
        qrs_width = int(0.1 * FS)
        sig_qrs = np.hanning(qrs_width) * np.random.uniform(0.8, 1.2)
        start_q = pos - qrs_width//2
        end_q   = start_q + qrs_width
        if end_q < N_SAMPLES:
            signal[:, start_q:end_q] += sig_qrs * np.random.uniform(0.5, 1.5, size=(N_LEADS, 1))
        
        # T wave
        t_pos = pos + int(0.3 * FS)
        t_width = int(0.2 * FS)
        sig_t = np.hanning(t_width) * np.random.uniform(0.3, 0.5)
        start_t = t_pos - t_width//2
        end_t   = start_t + t_width
        if end_t < N_SAMPLES:
            signal[:, start_t:end_t] += sig_t * np.random.uniform(0.5, 1.5, size=(N_LEADS, 1))

    # ── 2. Simulate Dissociated Atrial Rhythm (P-waves) ──────────────────────
    # Atrial rate is independent and faster (e.g., 70-100 bpm)
    a_rate = np.random.uniform(70, 100)
    a_interval = int((60 / a_rate) * FS)
    
    # Add random phase shift so P-waves land anywhere (including on T-waves)
    start_offset = np.random.randint(0, a_interval)
    a_positions = np.arange(start_offset, N_SAMPLES - a_interval//2, a_interval)
    
    for pos in a_positions:
        p_width = int(np.random.uniform(0.08, 0.12) * FS)
        sig_p = np.hanning(p_width) * np.random.uniform(0.1, 0.25) # Small amplitude
        start_p = pos - p_width//2
        end_p   = start_p + p_width
        
        if end_p < N_SAMPLES and start_p > 0:
            # Vary lead amplitudes, simulating physical heart vector
            lead_weights = np.random.uniform(0.1, 1.0, size=(N_LEADS, 1))
            signal[:, start_p:end_p] += sig_p * lead_weights
            
            # Save ground truth (onset, peak, offset)
            annotations['p_waves'].append((int(start_p), int(pos), int(end_p)))

    # ── 3. Add Physiologically Accurate Noise ────────────────────────────────
    time = np.arange(N_SAMPLES) / FS
    for l in range(N_LEADS):
        # 50Hz Powerline Interference
        power_noise = 0.02 * np.sin(2 * np.pi * 50 * time + np.random.rand())
        # Baseline Wander (low frequency < 0.5 Hz)
        wander = 0.1 * np.sin(2 * np.pi * np.random.uniform(0.1, 0.5) * time + np.random.rand())
        # EMG Noise (high frequency random noise)
        emg = np.random.normal(0, 0.01, N_SAMPLES)
        
        signal[l] += power_noise + wander + emg

    return signal, annotations


def generate_dataset():
    print(f"🧬 Booting Synthetic Generator...")
    print(f"   Target: {N_RECORDS} 12-lead ECG records.")
    print(f"   Constraints: 2nd/3rd Deg AV Block, Overlapping features.")
    
    all_signals = []
    all_annotations = []
    
    for i in range(N_RECORDS):
        sig, ann = generate_synthetic_record()
        all_signals.append(sig)
        all_annotations.append(ann)
        
        if (i+1) % 250 == 0:
            print(f"   Generated {i+1}/{N_RECORDS} records...")

    signals_path = os.path.join(OUTPUT_DIR, 'synthetic_signals.npy')
    annotations_path = os.path.join(OUTPUT_DIR, 'synthetic_annotations.pkl')
    
    np.save(signals_path, np.array(all_signals, dtype=np.float32))
    with open(annotations_path, 'wb') as f:
        pickle.dump(all_annotations, f)
        
    print(f"\n✅ Generation Complete!")
    print(f"   Signals stored at: {signals_path}")
    print(f"   Annotations stored at: {annotations_path}")


if __name__ == "__main__":
    generate_dataset()
