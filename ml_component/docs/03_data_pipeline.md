# 3. Data Pipeline — Loading, Augmentation, and Target Generation

This document provides a deep, line-by-line explanation of the three files responsible for converting raw PhysioNet ECG records into PyTorch-ready training tensors.

---

## 3.1 `download_data.py` — Dataset Acquisition

### Purpose
This script downloads the LUDB (Lobachevsky University Database) from the PhysioNet servers. It is the very first script a user runs.

### Code Breakdown

```python
DATA_DIR = os.path.join(os.getcwd(), 'data', 'raw', 'ludb')
os.makedirs(DATA_DIR, exist_ok=True)
```
**Why:** Creates the local storage folder. `exist_ok=True` prevents a crash if the folder already exists from a previous run.

```python
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
```
**Why:** The `wfdb.dl_database()` function prints extremely verbose download logs for every single file. This context manager temporarily redirects `stdout` to `/dev/null` (a black hole) so those logs are silently discarded, keeping the terminal clean. The original `stdout` is restored in `__exit__` to prevent permanent suppression.

```python
missing_records = []
for rec in records:
    if not (os.path.exists(...f"{rec}.hea") and os.path.exists(...f"{rec}.dat")):
        missing_records.append(rec)
```
**Why:** This is the **resume logic**. If the download fails halfway through (e.g., due to a PhysioNet `502 Bad Gateway` error), re-running the script will skip all previously downloaded records and only download the missing ones. Each LUDB record consists of two files: a `.hea` (header with metadata) and a `.dat` (binary signal data). Both must exist for a record to be considered complete.

```python
retries = 3
while retries > 0:
    try:
        wfdb.dl_database('ludb', dl_dir=DATA_DIR, records=[rec])
        break
    except Exception as e:
        retries -= 1
        time.sleep(2)
```
**Why:** PhysioNet is a free academic server that occasionally returns `502 Bad Gateway` errors under load. This retry mechanism waits 2 seconds and tries again, up to 3 times per record, before giving up. This prevents a single network hiccup from crashing the entire download.

---

## 3.2 `ludb_loader.py` — Raw Data Parsing

### Purpose
The `LUDBLoader` class reads the downloaded `.hea` and `.dat` files and extracts two things: the 12-lead ECG signal (a numerical array) and the P-wave annotations (onset/peak/offset coordinates).

### Class: `LUDBLoader`

#### Constructor (`__init__`)
```python
def __init__(self, data_dir, sampling_rate=500, target_length=5000):
    self.data_dir = Path(data_dir)
    self.sampling_rate = sampling_rate  # 500 Hz
    self.target_length = target_length  # 10 seconds × 500 Hz = 5000 samples
    self.records = self._get_record_list()
```
**Why:** The LUDB standard is 500 Hz sampling over 10 seconds. Setting `target_length=5000` ensures every record is exactly the same length, which is mandatory for batched neural network training (you cannot mix tensors of different lengths in a single batch without padding).

#### `_get_record_list()`
```python
hea_files = list(self.data_dir.rglob("*.hea"))
hea_files = [f for f in hea_files if ".ipynb_checkpoints" not in str(f)]
```
**Why:** `rglob("*.hea")` recursively searches for all header files. The filter removes any files accidentally created inside Jupyter checkpoint folders (a common contamination issue when notebooks are used alongside scripts).

#### `load_record(record_name)`
```python
record = wfdb.rdrecord(record_path)
signal = record.p_signal.T  # Shape: [12, 5000]
```
**Why:** `wfdb.rdrecord()` reads the binary `.dat` file and returns a Record object. The `.p_signal` attribute contains the physical signal values (in millivolts). It is originally shaped `[5000, 12]` (samples × leads), but we transpose it to `[12, 5000]` (leads × samples) because PyTorch's Conv1d expects the channel dimension first.

```python
if signal.shape[1] > self.target_length:
    signal = signal[:, :self.target_length]
elif signal.shape[1] < self.target_length:
    padding = self.target_length - signal.shape[1]
    signal = np.pad(signal, ((0,0), (0, padding)), mode='constant')
```
**Why:** Guarantees every signal is exactly 5000 samples. Records shorter than 10 seconds are zero-padded on the right. Records longer than 10 seconds are cropped. This uniform length is absolutely required for batch processing.

#### Annotation Parsing
```python
ann = wfdb.rdann(record_path, 'ii')  # Read Lead II annotations
```
**Why:** Lead II is the internationally standard lead for P-wave analysis because it provides the clearest view of atrial depolarization. The LUDB dataset stores per-lead annotations with extensions like `'ii'` for Lead II.

```python
if ann.symbol[i] == '(' and i+2 < len(ann.sample):
    if ann.symbol[i+1] == 'p' and ann.symbol[i+2] == ')':
        onset = ann.sample[i]
        peak = ann.sample[i+1]
        offset = ann.sample[i+2]
```
**Why:** The LUDB annotation format uses `(` for wave onset, a lowercase letter for the wave peak (e.g., `p` for P-wave, `t` for T-wave), and `)` for wave offset. This code searches for the specific triplet pattern `(`, `p`, `)` to extract complete P-wave instances. The `ann.sample` array gives the exact sample index (0–4999) where each annotation mark occurs.

#### `get_all_data()`
```python
if len(p_waves) > 0:  # Only keep records with P-wave labels
    all_signals.append(sig)
    all_annotations.append({'p_waves': p_waves})
```
**Why:** Some LUDB records may have missing or corrupted P-wave annotations (e.g., patients with Atrial Fibrillation where P-waves do not exist). These records are excluded because they provide no valid training targets and would inject noise into the loss function.

---

## 3.3 `augmentations.py` — Mathematical Noise Synthesis

### Purpose
This file implements **domain-specific** ECG augmentations adapted from the Joung et al. (2024) baseline paper. These augmentations add realistic, mathematically synthesized noise to real ECG signals during training to artificially multiply the dataset without generating fake heartbeats.

### Design Decision
**Why not use standard image augmentations (rotation, flipping, colour jitter)?** Because ECG signals are not images. Flipping an ECG horizontally would reverse time (making the heart contract backwards), and rotating it has no physical meaning. Instead, we simulate the exact types of noise that occur in real clinical environments.

### Function: `_tnoise_powerline()`
```python
def _tnoise_powerline(fs=500, N=5000, C=1.0, fn=50.0, K=3):
    t = torch.arange(0, N/fs, 1./fs)
    signal = torch.zeros(N)
    phi1 = random.uniform(0, 2*math.pi)
    for k in range(1, K+1):
        ak = random.uniform(0, 1)
        signal += C * ak * torch.cos(2*math.pi * k * fn * t + phi1)
    return signal * 0.05
```

**What it does:** Generates a synthetic 50 Hz interference signal (the frequency of electrical mains power in most countries).

**Mathematical formula:** `signal(t) = Σ aₖ · cos(2π · k · fₙ · t + φ₁)` for k = 1 to K

**Why each parameter exists:**
- `fs=500`: Matches the LUDB sampling rate (500 Hz).
- `N=5000`: Matches the signal length.
- `fn=50.0`: The base frequency of powerline noise (50 Hz in Europe/Asia, 60 Hz in Americas).
- `K=3`: Includes up to the 3rd harmonic (50, 100, 150 Hz). Real powerline noise contains harmonics.
- `phi1`: Random phase offset. Each augmented signal gets a different starting phase, creating unique noise patterns.
- `* 0.05`: Scales the noise to 5% of the signal amplitude. Too high would drown the ECG; too low would have no effect.

### Function: `_tnoise_baseline_wander()`
```python
def _tnoise_baseline_wander(fs=500, N=5000, C=1.0, fc=0.5):
```

**What it does:** Simulates the slow, low-frequency drift in ECG signals caused by patient breathing or electrode movement.

**Why `fc=0.5`:** Baseline wander is a very low-frequency phenomenon (below 0.5 Hz). A cutoff of 0.5 Hz means the synthetic wander completes roughly one full oscillation every 2 seconds, matching real respiratory artifacts.

### Class: `GaussianNoise`
```python
class GaussianNoise():
    def __init__(self, prob=0.3, scale=0.01):
    def __call__(self, wave):
        if random.random() < self.prob:
            wave += self.scale * torch.randn_like(wave)
        return wave
```

**What it does:** Adds random white noise to all leads with 30% probability.

**Why `scale=0.01`:** Clinical ECG amplitudes typically range from -1 to +1 mV after normalization. Adding noise at 1% of this range simulates realistic sensor noise without corrupting the underlying waveform morphology.

### Class: `BaselineShift`
```python
wave = wave + (self.scale * shift)
```

**What it does:** Shifts the entire signal up or down by a random constant. This simulates electrode impedance changes that cause DC offset drift.

### Class: `BaselineWander`
```python
wander = _tnoise_baseline_wander(fs=self.freq, N=len_wave)
wander = wander.repeat(channels, 1)  # Same wander on all 12 leads
wave = wave + wander
```

**Why identical wander on all leads:** Respiratory-induced baseline wander affects all electrodes simultaneously because the patient's chest is moving as a whole. Applying different random wander per lead would be physically unrealistic.

### Class: `PowerlineNoise`
```python
noise = _tnoise_powerline(fs=self.freq, N=len_wave)
noise = noise.repeat(channels, 1)
wave = wave + noise
```

**Why identical noise for all channels:** Powerline interference is an environmental electromagnetic effect that couples identically into all electrode wires.

### Factory Function: `get_research_augmentations()`
```python
def get_research_augmentations():
    return ECGCompose([
        BaselineWander(prob=0.3, freq=500),
        PowerlineNoise(prob=0.3, freq=500),
        GaussianNoise(prob=0.3, scale=0.01),
        BaselineShift(prob=0.3, scale=0.05),
    ])
```

**Why 30% probability for each:** Each augmentation is applied independently with 30% chance. This means roughly 30% of training samples get wander, 30% get powerline noise, etc. Some samples get multiple augmentations, some get none. This creates a rich diversity of training conditions while ensuring the model still sees plenty of clean signals.

---

## 3.4 `instance_dataset.py` — PyTorch Dataset & Target Generation

### Purpose
This is the central data interface. It wraps raw signals and annotations into a PyTorch `Dataset` that the `DataLoader` can iterate over during training. It handles three critical tasks: normalization, augmentation, and target tensor generation.

### Class: `AtrionInstanceDataset`

#### Constructor
```python
def __init__(self, signals, annotations, seq_len=5000, is_train=False):
    if self.is_train:
        self.baseline_augs = get_research_augmentations()
```
**Why conditional initialization:** Augmentations are only created when `is_train=True`. During validation and testing, the model must see clean, unmodified signals to produce fair, unbiased metrics.

#### `_normalize(sig)`
```python
def _normalize(self, sig):
    mu = np.mean(sig, axis=1, keepdims=True)
    sigma = np.std(sig, axis=1, keepdims=True) + 1e-6
    return (sig - mu) / sigma
```
**What it does:** Per-lead Z-score normalization. Each of the 12 leads is independently centered to mean=0 and scaled to standard deviation=1.

**Why per-lead:** Different ECG leads have vastly different amplitude ranges. Lead V1 might have amplitudes of ±0.5 mV while Lead II might have ±1.5 mV. Without per-lead normalization, the model would be biased towards high-amplitude leads.

**Why `+ 1e-6`:** Prevents division by zero for flat-line leads (e.g., a disconnected electrode where sigma=0).

#### `_augment(sig, centers, spans)`

**Step 1 — Noise Injection:**
```python
sig_tensor = torch.tensor(sig, dtype=torch.float32)
sig_tensor = self.baseline_augs(sig_tensor)
sig = sig_tensor.numpy()
```
Applies the Joung et al. augmentations (wander, powerline, etc.).

**Step 2 — Random Time Shift:**
```python
shift = np.random.randint(-250, 250)
```
**Why ±250 samples:** This is ±500ms at 500Hz. Shifting the signal left or right simulates slight timing variations in when the ECG recording started relative to the cardiac cycle.

**Step 3 — Amplitude Scaling:**
```python
sig = sig * np.random.uniform(0.8, 1.2)
```
**Why 0.8–1.2:** Simulates ±20% variation in electrode gain or patient body composition.

**Step 4 — Lead Dropout:**
```python
if np.random.rand() > 0.8:
    drop_indices = np.random.choice(12, size=np.random.randint(1, 3), replace=False)
    sig[drop_indices, :] = 0
```
**Why:** Simulates electrode disconnection (a common clinical artifact). With 20% probability, 1–2 random leads are zeroed out, forcing the model to learn P-wave detection from the remaining leads.

**Step 5 — Label Update:**
```python
new_c = c + shift
if 0 <= new_c < self.seq_len: new_centers.append(new_c)
```
**Critical:** When the signal is time-shifted, the P-wave annotations must be shifted by the exact same amount. If a P-wave center was at sample 2500 and the signal shifted right by 100 samples, the new center is at 2600. Centers that shift outside the signal boundaries are discarded.

#### `_generate_heatmap(centers, sigma=12)`
```python
def _generate_heatmap(self, centers, sigma=12):
    heatmap = np.zeros(self.seq_len, dtype=np.float32)
    x = np.arange(self.seq_len, dtype=np.float32)
    for center in centers:
        diff = (x - center)**2
        heatmap += np.exp(-(diff) / (2 * sigma**2))
    return np.clip(heatmap, 0.0, 1.0)
```

**What it does:** Generates a 1D Gaussian heatmap where each P-wave center produces a bell-shaped peak.

**Mathematical formula:** `H(x) = exp(-(x - c)² / (2σ²))` for each center c.

**Why Sigma=12:**
- Sigma=12 means the Gaussian peak spans approximately ±24 samples (±48ms at 500Hz) before dropping to near-zero.
- **Why not smaller (e.g., sigma=5)?** A sigma of 5 creates an extremely narrow target (only ~10 samples wide). On a sequence of 5000 samples, this is a 0.2% target area. The neural network's convolutional kernels must land almost exactly on the center to produce a non-zero gradient, causing severe vanishing gradient issues.
- **Why not larger (e.g., sigma=30)?** A sigma of 30 would cause neighbouring P-wave Gaussians to merge into each other, making it impossible to distinguish individual P-waves.
- Sigma=12 achieves the optimal balance: wide enough for stable gradient flow, narrow enough to preserve individual P-wave identity.

#### `__getitem__(idx)` — Target Tensor Construction
```python
heatmap[0] = self._generate_heatmap(centers)
```
Creates the Gaussian confidence target.

```python
width_map[0, int(center)] = (s_end - s_start) / self.seq_len
```
**What:** At exactly the center sample of each P-wave, stores the normalized width (duration / total_length). The width is stored as a fraction of the total sequence length to keep values between 0 and 1.

**Why only at the center:** The width is a property of the entire P-wave, not of individual samples. Storing it only at the center avoids ambiguity about which samples "own" the width value.

```python
mask[0, int(s_start):int(s_end)] = 1.0
```
**What:** A binary mask where all samples belonging to any P-wave are set to 1, and everything else is 0. This provides auxiliary spatial information that helps the model learn the boundaries of P-waves.
