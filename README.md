# Eyes_on_falls_YOLO

# 🎥 Distributed Fall Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/)
[![Raspberry Pi](https://img.shields.io/badge/platform-Raspberry%20Pi%204-red.svg)](https://www.raspberrypi.org/)
[![Status](https://img.shields.io/badge/status-Research%20Prototype-orange.svg)]()

A real-time, distributed fall detection system using multiple RGB cameras, neural networks, and adaptive consensus algorithms. Designed for elderly care facilities (RSA) with focus on reliability, low latency, and bias prevention.

## 🌟 Key Features

- **Multi-camera consensus**: 3+ cameras collaborate to make robust decisions
- **Real-time performance**: <25ms decision latency @ 30 FPS
- **Adaptive consensus**: Combines RAFT, Gossip Protocol, and Weighted Voting
- **Byzantine fault tolerance**: Handles malfunctioning cameras gracefully
- **Anti-bias mechanisms**: Prevents positional and temporal biases
- **Edge computing**: Runs entirely on Raspberry Pi devices

## 📋 Table of Contents

- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Consensus Algorithms](#consensus-algorithms)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Academic References](#academic-references)

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Camera A   │     │  Camera B   │     │  Camera C   │
│  (Ceiling)  │     │  (Corner L) │     │  (Corner R) │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│Raspberry Pi │     │Raspberry Pi │     │Raspberry Pi │
│     A       │◄───►│     B       │◄───►│     C       │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       └───────────────────┴───────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Alert     │
                    │   System    │
                    └─────────────┘
```

### Components

- **Neural Network**: MobileNetV2-based fall detection model
- **Consensus Layer**: Distributed decision-making system
- **Communication**: TCP/IP sockets with MessagePack serialization
- **Monitoring**: Real-time dashboard for system health

## 💻 Requirements

### Hardware
- 3x Raspberry Pi 4 (8GB RAM recommended)
- 3x USB/CSI RGB Cameras (minimum 720p @ 30fps)
- Gigabit Ethernet Switch
- Reliable power supply (3x 15W)

### Software
- Raspberry Pi OS (64-bit)
- Python 3.8+
- TensorFlow Lite 2.10+
- OpenCV 4.5+
- NumPy, MessagePack, AsyncIO

## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/distributed-fall-detection.git
cd distributed-fall-detection
```

### 2. Install Dependencies
```bash
# On each Raspberry Pi
sudo apt-get update
sudo apt-get install -y python3-pip python3-opencv python3-numpy

pip3 install -r requirements.txt
```

### 3. Deploy Model
```bash
# Copy the trained model to each Pi
scp models/fall_detection_mobilenet.tflite pi@192.168.1.10:/home/pi/fall-detection/models/
```

### 4. Configure Network
Edit `config/network.yaml`:
```yaml
nodes:
  camera_a:
    ip: "192.168.1.10"
    port: 5555
    position: "ceiling"
  camera_b:
    ip: "192.168.1.11"
    port: 5555
    position: "corner_left"
  camera_c:
    ip: "192.168.1.12"
    port: 5555
    position: "corner_right"

consensus:
  algorithm: "adaptive"  # adaptive, raft, weighted
  gossip_interval_ms: 50
  decision_timeout_ms: 20
  leader_rotation_frames: 1000
```

## 🎮 Usage

### Starting the System

On each Raspberry Pi:
```bash
# Node A (initial leader)
python3 main.py --node-id camera_a --config config/network.yaml

# Node B
python3 main.py --node-id camera_b --config config/network.yaml

# Node C
python3 main.py --node-id camera_c --config config/network.yaml
```

### Monitoring Dashboard
```bash
# On monitoring station
python3 dashboard.py --config config/network.yaml
```

### Testing
```bash
# Run unit tests
python3 -m pytest tests/

# Run consensus simulation
python3 tools/simulate_consensus.py --scenarios tests/scenarios/
```

## 🧮 Consensus Algorithms

### Weighted Consensus Formula
```
Decision = 1 if Σ(wi × Ci × Di) / Σ(wi × Ci) ≥ θ
```
Where:
- `wi`: Historical weight of camera i
- `Ci`: Confidence of current detection
- `Di`: Detection decision (0 or 1)
- `θ`: Adaptive threshold (default 0.6)

### Weight Evolution
```
wi(t+1) = α × wi(t) + (1-α) × ri(t)
```
- `α`: Memory factor (0.85)
- `ri`: Reward (1 if correct, 0 if wrong)

### Leader Selection Score
```
score = 0.5 × accuracy + 0.3 × stability + 0.2 × speed
```

## 📊 Performance

### Latency Breakdown
| Component | Time |
|-----------|------|
| Frame capture | ~5ms |
| NN inference | 10-12ms |
| Consensus | 3-5ms |
| Communication | <2ms |
| **Total** | **<25ms** |

### Accuracy Metrics
- **Single Camera**: 91.3% F1-score
- **3-Camera Consensus**: 97.2% F1-score
- **With 1 Camera Offline**: 95.1% F1-score

### Resource Usage
- CPU: ~60% per Pi during inference
- RAM: ~500MB per node
- Network: <1Mbps between nodes
- Power: 15W per node

## 🔧 Troubleshooting

### Common Issues

**High Latency**
```bash
# Check network latency
ping -c 100 192.168.1.11

# Verify gossip frequency
tail -f logs/gossip.log | grep "interval"
```

**Camera Offline**
```bash
# Check camera status
python3 tools/check_camera.py --node-id camera_b

# Force recalibration
python3 tools/recalibrate.py --node-id camera_b
```

**Byzantine Behavior**
```bash
# View deviation metrics
python3 tools/analyze_bias.py --logs logs/

# Reset node weights
python3 tools/reset_weights.py --node-id camera_c --soft
```

### Development Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -r requirements-dev.txt

# Run linters
black src/
flake8 src/
mypy src/
```

## 📚 Academic References

This project implements concepts from:

1. **Consensus Algorithms**
   - Ongaro, D., & Ousterhout, J. (2014). "In Search of an Understandable Consensus Algorithm (RAFT)"
   - Lamport, L. (1998). "The Part-Time Parliament (Paxos)"

2. **Distributed Systems**
   - Van Renesse, R., et al. (1998). "A Gossip-Style Failure Detection Service"
   - Castro, M., & Liskov, B. (1999). "Practical Byzantine Fault Tolerance"

3. **Fall Detection**
   - Kwolek, B., & Kepski, M. (2014). "Human fall detection on embedded platform using depth maps and wireless accelerometer"

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Prof. Aldo [Surname] for distributed systems guidance
- RSA partners in Abruzzo for real-world testing
- Open source community for foundational libraries

---

**Note**: This is a research prototype. For production deployment in healthcare settings, additional certifications and safety validations are required.
