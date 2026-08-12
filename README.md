# C++ autoencoder: training and prediction

This directory provides two C++17/LibTorch programs:

| Program | Purpose | Sources |
|---|---|---|
| `autoencoder_cpp` | Train and save the metric autoencoder | `main.cpp`, `model.cpp`, `model.h` |
| `usage_cpp` | Load the saved model and predict a peptide path | `usage.cpp`, `model.cpp`, `model.h` |

`autoencoder_cpp` is the C++ port of `new_model.py`. `usage_cpp` is the C++
replacement for `usage.py` and consumes the artifacts created by the trainer.

## Verified environment

The build and integration test were verified on Fedora Linux x86_64 with
Python 3.13.14, CPU PyTorch/LibTorch 2.13.0, GCC 15.2, CMake 4.4, and C++17.
The checked-in `cpp/libtorch/` is PyTorch 1.2 from 2019 and is incompatible
with current GCC; do not select it.

The repository is on a removable filesystem that may reject symlinks or be
mounted `noexec`. The setup script therefore puts its virtual environment and
compiled binaries in `/tmp`.

## 1. Install, build, and test automatically

Run from the repository root:

```bash
SKIP_SYSTEM_PACKAGES=1 bash cpp/setup_environment.sh
```

Remove `SKIP_SYSTEM_PACKAGES=1` on a new machine where the compiler, CMake,
Python, and pip still need to be installed:

```bash
bash cpp/setup_environment.sh
```

The script:

1. Optionally installs system build tools using `dnf` or `apt-get`.
2. Creates `/tmp/autoencoder-cpp-$USER-venv`.
3. installs CPU-only PyTorch without caching CUDA packages.
4. Builds both programs in `/tmp/autoencoder-cpp-$USER-build`.
5. Runs the complete train-save-load-predict integration test.

It prefers Python 3.13 when available. Optional overrides are:

```bash
PYTHON_BIN=python3.13 \
VENV_DIR=/tmp/my-autoencoder-venv \
BUILD_DIR=/tmp/my-autoencoder-build \
bash cpp/setup_environment.sh
```

## 2. Build manually

```bash
python3.13 -m venv /tmp/autoencoder-cpp-$USER-venv
source /tmp/autoencoder-cpp-$USER-venv/bin/activate
python -m pip install --no-cache-dir --upgrade pip wheel
python -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu torch

cmake -S cpp -B /tmp/autoencoder-cpp-$USER-build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/autoencoder-cpp-$USER-build -j1
```

`-j1` reduces peak memory use when compiling LibTorch headers. Build only one
program when desired:

```bash
cmake --build /tmp/autoencoder-cpp-$USER-build \
  --target autoencoder_cpp -j1

cmake --build /tmp/autoencoder-cpp-$USER-build \
  --target usage_cpp -j1
```

The resulting programs are:

```text
/tmp/autoencoder-cpp-$USER-build/autoencoder_cpp
/tmp/autoencoder-cpp-$USER-build/usage_cpp
```

## 3. Training CSV format

The trainer accepts a directory and reads every `.csv` file directly inside it
in sorted filename order. Every file in that directory must have these columns:

```csv
SS,MM,GT
A,10.0,1.0
B,12.0,1.5
A,15.0,2.5
B,19.0,4.0
```

- `SS`: monomer/symbol added at the row.
- `MM`: mass measurement.
- `GT`: ground-truth or retention-time measurement.

Every row after the first creates a six-value transition: previous mass/GT,
current mass/GT, mass difference, and GT difference. Keep unrelated CSV files,
especially inference peak files, outside the training directory.

The small test training set is:

```text
cpp/tests/data/simple/transitions.csv
```

## 4. Train with `autoencoder_cpp`

Syntax:

```text
autoencoder_cpp TRAINING_CSV_FOLDER [EPOCHS] [BATCH_SIZE]
```

- `EPOCHS` is the number of complete passes over the training set; default 200.
- `BATCH_SIZE` is the number of transitions per optimizer update; default 64.

Run from a separate output directory because artifacts are written to the
current working directory:

```bash
PROJECT_ROOT="$PWD"
mkdir -p /tmp/autoencoder-run
cd /tmp/autoencoder-run

/tmp/autoencoder-cpp-$USER-build/autoencoder_cpp \
  "$PROJECT_ROOT/cpp/tests/data/simple" 200 64
```

This creates:

```text
/tmp/autoencoder-run/autoencoder_model.pt
/tmp/autoencoder-run/norm.txt
```

- `autoencoder_model.pt`: trained network weights and module archive.
- `norm.txt`: six feature means followed by six standard deviations.

For real data, replace the test folder with your dedicated training directory:

```bash
/tmp/autoencoder-cpp-$USER-build/autoencoder_cpp \
  /absolute/path/to/my-training-data 200 64
```

## 5. Input CSV format for `usage_cpp`

The prediction input is one CSV file with this schema:

```csv
MM,GG
10.0,1.0
11.0,1.2
12.0,1.5
15.0,2.5
```

The peak file can include noise peaks. Keep it outside the training CSV folder.
A working example is `cpp/tests/peaks.csv`.

## 6. Predict with `usage_cpp`

Syntax:

```text
usage_cpp MODEL NORM TRAINING_CSV_FOLDER PEAKS_CSV [INITIAL_MONOMER]
```

Run the prediction using the model produced in the previous step:

```bash
/tmp/autoencoder-cpp-$USER-build/usage_cpp \
  /tmp/autoencoder-run/autoencoder_model.pt \
  /tmp/autoencoder-run/norm.txt \
  "$PROJECT_ROOT/cpp/tests/data/simple" \
  "$PROJECT_ROOT/cpp/tests/input.csv" \
  A
```

`A` is the optional known first monomer. Omit it when the first species is not
known.

The training CSV folder should match the data used to train the model. The model
archive contains network weights but not label names, latent class centroids,
or physical mass/RT statistics. `usage_cpp` reconstructs those values from the
reference training folder and then:

1. Normalizes each candidate transition with `norm.txt`.
2. Encodes it using the loaded model.
3. Scores latent similarity, mass accuracy, and RT consistency.
4. Searches for the best peptide path while skipping unlikely/noise peaks.
5. Prints the predicted sequence, selected peaks, transition scores, and path
   score.

## 7. Complete example

From the repository root:

```bash
PROJECT_ROOT="$PWD"
BUILD_DIR="/tmp/autoencoder-cpp-$USER-build"
RUN_DIR=/tmp/autoencoder-run
TRAIN_DIR="$PROJECT_ROOT/cpp/tests/data/simple"
PEAKS_FILE="$PROJECT_ROOT/cpp/tests/peaks.csv"

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

"$BUILD_DIR/autoencoder_cpp" "$TRAIN_DIR" 200 64

"$BUILD_DIR/usage_cpp" \
  "$RUN_DIR/autoencoder_model.pt" \
  "$RUN_DIR/norm.txt" \
  "$TRAIN_DIR" \
  "$PEAKS_FILE" \
  A
```

## 8. Integration test

```bash
ctest --test-dir /tmp/autoencoder-cpp-$USER-build --output-on-failure
```

The test performs the entire workflow:

1. Trains with `autoencoder_cpp`.
2. Confirms a 2D embedding was generated.
3. Verifies `autoencoder_model.pt` and all 12 normalization values.
4. Loads those files with `usage_cpp`.
5. Confirms that prediction output is produced.

Test artifacts are kept in:

```text
/tmp/autoencoder-cpp-$USER-build/integration-output
```

## Troubleshooting

- `No such file or directory`: run the setup/build command and use the binary
  under `/tmp/autoencoder-cpp-$USER-build`.
- `Permission denied` for a binary inside the repository: the removable drive
  is mounted `noexec`; build and run from `/tmp`.
- `No space left on device` while installing `nvidia-*`: use the corrected
  setup script, which installs from the CPU-only PyTorch index with
  `--no-cache-dir`.
- `CSV missing required columns`: the training folder contains a non-training
  CSV. Move the peaks or other CSV files to another directory.
- `library kineto not found`: this is an optional profiler warning and does not
  prevent this CPU build from compiling or running.
