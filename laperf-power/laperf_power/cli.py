#!/usr/bin/env python3
"""Real-time power monitoring CLI tool.

Usage:
    laperf-power [--interval SECONDS] [--no-sudo] [--output FILE]

Press Ctrl+C to stop monitoring and see final statistics.
"""

import argparse
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from laperf_power.power_metrics import PowerMetrics


class PowerMonitorCLI:
    """Interactive CLI for real-time power monitoring."""

    def __init__(
        self,
        interval: float = 1.0,
        use_sudo: bool = True,
        output_file: Path | None = None,
    ):
        self.interval = interval
        self.output_file = output_file
        self.monitor = PowerMetrics(
            interval=interval, use_sudo_powermetrics=use_sudo
        )
        self.running = False

        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\n⏹  Stopping monitoring...")
        self.running = False

    def _print_header(self):
        """Print monitoring header."""
        print("\n" + "=" * 80)
        print("⚡ REAL-TIME POWER MONITORING")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Interval: {self.interval}s")
        if self.output_file:
            print(f"Output: {self.output_file}")
        print("=" * 80)
        print("\nPress Ctrl+C to stop and view statistics\n")

    def _print_live_metrics(self, sample_count: int):
        """Print current metrics in real-time."""
        # Get latest values from each metric
        latest = {}
        for key, values in self.monitor.metrics.items():
            if values:
                latest[key] = values[-1]

        # Clear line and print metrics
        print(f"\r[Sample #{sample_count}] ", end="")

        metrics_to_show = []

        # GPU metrics
        if "gpu_watts" in latest:
            metrics_to_show.append(f"GPU: {latest['gpu_watts']:.1f}W")
        if "gpu_util_percent" in latest:
            metrics_to_show.append(f"{latest['gpu_util_percent']:.0f}%")
        if "gpu_vram_used_mb" in latest:
            vram_used = latest['gpu_vram_used_mb'] / 1024
            metrics_to_show.append(f"{vram_used:.1f}GB")

        # CPU metrics
        if "cpu_util_percent" in latest:
            metrics_to_show.append(f"CPU: {latest['cpu_util_percent']:.0f}%")
        if "cpu_watts" in latest:
            metrics_to_show.append(f"{latest['cpu_watts']:.1f}W")

        # RAM
        if "ram_used_gb" in latest:
            metrics_to_show.append(f"RAM: {latest['ram_used_gb']:.1f}GB")

        # System power
        if "system_watts" in latest:
            metrics_to_show.append(f"SYS: {latest['system_watts']:.1f}W")

        # Temperature
        if "gpu_temp_celsius" in latest:
            metrics_to_show.append(f"Temp: {latest['gpu_temp_celsius']:.0f}°C")

        print(" | ".join(metrics_to_show), end="", flush=True)

    def _print_summary(self, results: dict):
        """Print final statistics summary."""
        print("\n\n" + "=" * 80)
        print("📊 MONITORING SUMMARY")
        print("=" * 80)

        # Duration and samples
        print(f"\nDuration: {results.get('monitoring_duration_seconds', 0):.1f}s")
        print(f"Samples collected: {results.get('samples_collected', 0)}")

        # GPU Power
        if "gpu_watts_p50" in results:
            print("\n🎮 GPU Power:")
            print(f"  P50: {results['gpu_watts_p50']:.2f}W")
            print(f"  P95: {results['gpu_watts_p95']:.2f}W")

        # CPU Power
        if "cpu_watts_p50" in results:
            print("\n💻 CPU Power:")
            print(f"  P50: {results['cpu_watts_p50']:.2f}W")
            print(f"  P95: {results['cpu_watts_p95']:.2f}W")

        # System Power
        if "system_watts_p50" in results:
            print("\n⚡ System Power:")
            print(f"  P50: {results['system_watts_p50']:.2f}W")
            print(f"  P95: {results['system_watts_p95']:.2f}W")

        # GPU Utilization
        if "gpu_util_percent_p50" in results:
            print("\n🎯 GPU Utilization:")
            print(f"  P50: {results['gpu_util_percent_p50']:.0f}%")
            print(f"  P95: {results['gpu_util_percent_p95']:.0f}%")

        # GPU VRAM
        if "gpu_vram_used_mb_p50" in results:
            vram_p50 = results['gpu_vram_used_mb_p50'] / 1024
            vram_p95 = results['gpu_vram_used_mb_p95'] / 1024
            print("\n💾 GPU VRAM:")
            print(f"  P50: {vram_p50:.2f}GB")
            print(f"  P95: {vram_p95:.2f}GB")

        # CPU Utilization
        if "cpu_util_percent_p50" in results:
            print("\n🔧 CPU Utilization:")
            print(f"  P50: {results['cpu_util_percent_p50']:.0f}%")
            print(f"  P95: {results['cpu_util_percent_p95']:.0f}%")

        # RAM
        if "ram_used_gb_p50" in results:
            print("\n🧠 RAM Usage:")
            print(f"  P50: {results['ram_used_gb_p50']:.2f}GB")
            print(f"  P95: {results['ram_used_gb_p95']:.2f}GB")

        # GPU Temperature
        if "gpu_temp_celsius_p50" in results:
            print("\n🌡️  GPU Temperature:")
            print(f"  P50: {results['gpu_temp_celsius_p50']:.0f}°C")
            print(f"  P95: {results['gpu_temp_celsius_p95']:.0f}°C")

        # Battery
        if "battery_drain_percent" in results:
            print("\n🔋 Battery:")
            print(
                f"  Start: {results['battery_start_percent']:.1f}% → "
                f"End: {results['battery_end_percent']:.1f}%"
            )
            print(f"  Drain: {results['battery_drain_percent']:.1f}%")

        print("\n" + "=" * 80 + "\n")

    def _save_results(self, results: dict):
        """Save results to JSON file."""
        if not self.output_file:
            return

        try:
            # Add timestamp
            results["timestamp"] = datetime.now().isoformat()

            with open(self.output_file, "w") as f:
                json.dump(results, f, indent=2)

            print(f"✓ Results saved to: {self.output_file}")
        except Exception as e:
            print(f"⚠️  Failed to save results: {e}")

    def run(self):
        """Run the monitoring loop."""
        self._print_header()

        # Start monitoring
        self.monitor.start()
        self.running = True

        sample_count = 0

        try:
            while self.running:
                # Wait for next sample
                time.sleep(self.interval)

                # Print live metrics
                sample_count = len(self.monitor.timestamps)
                if sample_count > 0:
                    self._print_live_metrics(sample_count)

        except KeyboardInterrupt:
            # This shouldn't happen due to signal handler, but just in case
            pass

        # Stop monitoring and get results
        print("\n\nCollecting final statistics...")
        results = self.monitor.stop()

        # Print summary
        self._print_summary(results)

        # Save to file if requested
        if self.output_file:
            self._save_results(results)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time power and resource monitoring tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor with default settings (10s interval, sudo enabled)
  laperf-power

  # Monitor with custom interval (faster sampling)
  laperf-power --interval 1.0

  # Monitor without sudo (no detailed power metrics)
  laperf-power --no-sudo

  # Save results to file
  laperf-power --output results.json

Press Ctrl+C to stop monitoring and view statistics.
        """,
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="Sampling interval in seconds (default: 10.0)",
    )

    parser.add_argument(
        "--no-sudo",
        action="store_true",
        help="Disable sudo powermetrics (no detailed GPU/CPU power on macOS)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        metavar="FILE",
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    # Validate interval
    if args.interval <= 0:
        print("Error: Interval must be greater than 0")
        sys.exit(1)

    # Create and run monitor
    cli = PowerMonitorCLI(
        interval=args.interval,
        use_sudo=not args.no_sudo,
        output_file=args.output,
    )

    cli.run()


if __name__ == "__main__":
    main()
