![banner](https://github.com/sleepy-wood/ml-sleep/blob/main/ml-sleep.png)

# ml-sleep

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![MIT License][license-shield]][license-url]

<div align="center">
  <a href="https://github.com/sleepy-wood">
    <img src="https://github.com/sleepy-wood/client-web/blob/dev/src/assets/images/logo.png" alt="Logo" width="120" height="120">
  </a>
  <h3 align="center">Realtime Sleep Detection from Apple Watch Sensors</h3>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#install-dependencies">Install Dependencies</a></li>
        <li><a href="#train-the-model">Train the Model</a></li>
        <li><a href="#export-to-the-coreml-model">Export to the CoreML Model</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

-   [미시간 대학교 애플 워치와 수면 다원 검사 데이터](https://physionet.org/content/sleep-accel/1.0.0/)
-   CoreML Model로 Export
-   CUDA 이외에도 Apple Silicon 기기에서도 학습 가능 (MPS backend)

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

1. Download Data

위 데이터 링크에서 데이터셋 다운로드 후 `data` 폴더로 압축 해제 (`data/heart_rate` ... 형태가 되도록)

2. Create & Activate Virtualenv
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```

### Install Dependencies

```bash
pip install -e .[dev]
```

### Train the Model

```bash
cli fit -c config.yaml
```

### Export to the CoreML Model

[notebooks/coreml.ipynb](notebooks/coreml.ipynb) 참조

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

[contributors-shield]: https://img.shields.io/github/contributors/sleepy-wood/ml-sleep.svg?style=for-the-badge
[contributors-url]: https://github.com/sleepy-wood/ml-sleep/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/sleepy-wood/ml-sleep.svg?style=for-the-badge
[forks-url]: https://github.com/sleepy-wood/ml-sleep/network/members
[stars-shield]: https://img.shields.io/github/stars/sleepy-wood/ml-sleep.svg?style=for-the-badge
[stars-url]: https://github.com/sleepy-wood/ml-sleep/stargazers
[license-shield]: https://img.shields.io/github/license/sleepy-wood/ml-sleep.svg?style=for-the-badge
[license-url]: https://github.com/sleepy-wood/ml-sleep/blob/master/LICENSE.txt
