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

### Introducing

- [미시간 대학교 애플 워치와 수면 다원 검사 데이터](https://physionet.org/content/sleep-accel/1.0.0/)
- CoreML Model로 Export
- CUDA 이외에도 Apple Silicon 기기에서도 학습 가능 (MPS backend)

### 실시간 수면 탐지를 위한 AI 모델 학습 및 배포

- 애플에서 제공하는 수면 분석은 기상 시간 이후에 한꺼번에 배치(Batch)로 수면 시간 동안의 데이터를 분석하여 수면 여부와 단계를 제공한다. 따라서 수면 중에는 수면 여부가 제공되지 않으므로 수면 중에 수면 여부를 알 수 없다. 이는 이 프로젝트에서 지향하는 ‘실시간 현실 반영’ 메타버스의 취지에 부합하지 않는다는 판단 하에, 실시간 수면 탐지 AI 모델을 제작하고 수면 중인 있는 사용자의 나무에 수면 중임을 나타내는 독특한 이펙트를 부여하기로 하였다. 이를 위해 미시간 대학교에서 측정한 애플 워치와 수면 다원 검사 31명분의 데이터(Walch, O., 2019))를 이용하기로 하였고, 약 89.0%의 검증 정확도(Validation Accuracy)를 나타내었다.
- 실제 적용 시에는 애플에서 센서 데이터로부터 사용자의 움직임 상태를 실시간으로 분석해서 제공하는 API도 함께 활용하여 정확도를 더욱 보정하였다. 예를 들어, 이러한 분석에서 제공하는 것 중에 하나가 자전거를 타고 있는지의 여부인데, 자전거를 타면서 자는 사람은 사실상 없을 것이다.
- 학습한 모델은 어디까지나 보조적인 용도로, 수면 중에 수면 여부를 탐지하고 메타버스 상에서 이를 반영하기 위함이지 건강 분석에 사용되지는 않을 예정이다. 실제로 건강 분석은 실시간일 필요가 없으므로 더 많은 데이터에서 검증된, 애플이 기상 시간 이후에 한꺼번에 분석해 제공한 수면 데이터를 활용하여 건강을 판단할 것이다. 
- 이 모델은 Pytorch로 학습된 뒤 CoreML Model로 변환된 후 Neural Engine을 통해 클라이언트 사이드에서 가속된다. 유니티에서 이러한 신경망 가속 하드웨어를 호출할 수 있는 API가 없어 직접 제작하였다. 

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
