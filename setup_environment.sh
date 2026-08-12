#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif command -v python3.13 >/dev/null 2>&1; then
  PYTHON_BIN="python3.13"
else
  PYTHON_BIN="python3"
fi
VENV_DIR="${VENV_DIR:-/tmp/autoencoder-cpp-${USER:-user}-venv}"
BUILD_DIR="${BUILD_DIR:-/tmp/autoencoder-cpp-${USER:-user}-build}"

run_as_admin() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "Administrator access is required to install system packages." >&2
    echo "Install them manually, then rerun with SKIP_SYSTEM_PACKAGES=1." >&2
    exit 1
  fi
}

install_system_packages() {
  if command -v dnf >/dev/null 2>&1; then
    run_as_admin dnf install -y gcc-c++ cmake python3 python3-pip python3-devel
  elif command -v apt-get >/dev/null 2>&1; then
    run_as_admin apt-get update
    run_as_admin apt-get install -y build-essential cmake python3 python3-pip python3-venv python3-dev
  else
    echo "Unsupported package manager. Install a C++17 compiler, CMake 3.18+," >&2
    echo "Python 3, pip, and Python venv support, then rerun with" >&2
    echo "SKIP_SYSTEM_PACKAGES=1." >&2
    exit 1
  fi
}

if [[ "${SKIP_SYSTEM_PACKAGES:-0}" != "1" ]]; then
  install_system_packages
fi

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
}

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --no-cache-dir --upgrade pip wheel
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

python - <<'PY_CHECK'
import torch
print(f"Using PyTorch {torch.__version__}")
print(f"LibTorch CMake prefix: {torch.utils.cmake_prefix_path}")
print(f"C++11 ABI: {torch.compiled_with_cxx11_abi()}")
PY_CHECK

cmake -S "${PROJECT_ROOT}/cpp" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j1
ctest --test-dir "${BUILD_DIR}" --output-on-failure

printf '\nEnvironment ready. Activate it later with:\n  source %s/bin/activate\n' "${VENV_DIR}"
printf 'Run the example with:\n  %s/autoencoder_cpp %s/cpp/tests/data 10 3\n' "${BUILD_DIR}" "${PROJECT_ROOT}"
