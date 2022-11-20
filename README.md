# Realtime Sleep Detection from iPhone & Apple Watch Sensors

-   [미시간 대학교 애플 워치와 수면 다원 검사 데이터](https://physionet.org/content/sleep-accel/1.0.0/)
-   CoreML Model로 Export
-   CUDA 이외에도 Apple Silicon 기기에서도 학습 가능 (MPS backend)

## Procedure

### 0. Download Data

위 데이터 링크데이터에서 데이터셋 다운로드 후 `data` folder로 압축 해제 (`data/heart_rate` ... 형태가 되도록)

### 1. Create & Activate Virtualenv

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -e .[dev]
```

### 3. Train the Model

```bash
cli fit -c config.yaml
```

### 4. Export to the CoreML Model

[notebooks/coreml.ipynb](notebooks/coreml.ipynb) 참조
