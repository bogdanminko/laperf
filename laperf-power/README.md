# LaPerf Power

Lightweight real-time power and resource monitoring tool for AI/ML workloads.

## Features

- **Real-time monitoring** with live console output
- **Cross-platform support**: macOS (Apple Silicon + Intel), Linux (NVIDIA), Windows
- **Comprehensive metrics**: GPU/CPU power, utilization, VRAM, temperature, RAM, battery
- **Statistical analysis**: P50/P95 percentiles for all metrics
- **JSON export** for further analysis
- **Zero heavy dependencies**: Only requires `psutil` (~5 MB installed)

## Installation

### Quick run without installation (recommended)

```bash
uvx laperf-power
```

### Install as a global tool

```bash
# Using uv
uv tool install laperf-power

# Using pip
pip install laperf-power
```

## Usage

```bash
# Start monitoring with default settings (10s interval)
laperf-power

# Custom sampling interval (faster sampling = more overhead)
laperf-power --interval 1.0

# Without sudo (basic metrics only on macOS)
laperf-power --no-sudo

# Save results to JSON
laperf-power --output power_metrics.json
```

**Press Ctrl+C to stop and view statistics.**

## What It Monitors

### GPU Metrics
- **Power consumption** (Watts) - NVIDIA via `nvidia-smi`, macOS via `powermetrics`
- **Utilization** (%) - GPU compute usage
- **VRAM** (GB) - Memory used/total
- **Temperature** (°C) - Die temperature

### CPU Metrics
- **Power consumption** (Watts) - macOS only with sudo
- **Utilization** (%) - Average across all cores

### System Metrics
- **RAM usage** (GB) - Process memory consumption
- **Battery drain** (%) - Change during monitoring period

## Platform Support

| Platform | GPU Power | GPU Stats | CPU Power | Notes |
|----------|-----------|-----------|-----------|-------|
| **macOS (Apple Silicon)** | ✅ (with sudo) | ✅ | ✅ (with sudo) | Via `powermetrics` and `ioreg` |
| **macOS (Intel)** | ❌ | ❌ | ✅ (with sudo) | Via `powermetrics` |
| **Linux (NVIDIA)** | ✅ | ✅ | ❌ | Via `nvidia-smi` |
| **Linux (AMD/Intel)** | ❌ | ❌ | ❌ | CPU/RAM only |
| **Windows** | ❌ | ❌ | ❌ | CPU/RAM only |

## Example Output

```
⚡ REAL-TIME POWER MONITORING
================================================================================
Started: 2025-11-27 14:30:00
Interval: 10.0s
================================================================================

Press Ctrl+C to stop and view statistics

[Sample #42] GPU: 11.7W 32% 8.2GB | CPU: 15% 1.0W | RAM: 16.3GB | Temp: 45°C
```

**Final statistics:**

```
📊 MONITORING SUMMARY
================================================================================

Duration: 420.0s
Samples collected: 42

🎮 GPU Power:
  P50: 11.7W
  P95: 13.2W

💻 CPU Power:
  P50: 1.0W
  P95: 1.5W

🎯 GPU Utilization:
  P50: 32%
  P95: 45%

💾 GPU VRAM:
  P50: 8.2GB
  P95: 8.5GB

🔧 CPU Utilization:
  P50: 15%
  P95: 22%

🧠 RAM Usage:
  P50: 16.3GB
  P95: 16.7GB

🔋 Battery:
  Start: 85.0% → End: 83.5%
  Drain: 1.5%
```

## macOS sudo Setup (Optional)

For detailed GPU/CPU power metrics on macOS, `laperf-power` uses `sudo powermetrics`.

**Option 1: Enter password when prompted** (recommended for occasional use)

**Option 2: Passwordless sudo** (for frequent use)

Add to `/etc/sudoers` (use `sudo visudo`):
```
your_username ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
```

## Use Cases

- **AI/ML Development**: Monitor power usage during model training/inference
- **Hardware Evaluation**: Compare power efficiency across different GPUs
- **Performance Optimization**: Identify power/performance bottlenecks
- **Battery Life Testing**: Track power consumption on laptops
- **System Monitoring**: Real-time resource usage dashboard

## Part of LaPerf

`laperf-power` is extracted from [LaPerf](https://github.com/bogdanminko/laperf) - a comprehensive AI hardware benchmark suite. For full benchmarking capabilities (embeddings, LLMs, VLMs), check out the main project.

## License

Apache-2.0

## Contributing

Issues and PRs welcome at [https://github.com/bogdanminko/laperf](https://github.com/bogdanminko/laperf)
