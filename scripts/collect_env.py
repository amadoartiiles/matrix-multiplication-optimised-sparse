"""Record the hardware and software environment of a benchmark run.

    python scripts/collect_env.py [--out results/environment.json]
"""
import argparse
import json
import os
import platform
import subprocess
import sys

import psutil


def cmd_output(cmd):
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (res.stdout + res.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def powershell(expr):
    return cmd_output(["powershell", "-NoProfile", "-Command", expr])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/environment.json")
    parser.add_argument("--config", default="config/benchmark_config.json")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    battery = psutil.sensors_battery()
    env = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_processors": psutil.cpu_count(logical=True),
        "ram_bytes": psutil.virtual_memory().total,
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "psutil": psutil.__version__,
        "gcc_version": cmd_output([os.environ.get("CC", cfg["c"]["compiler"]), "--version"]).splitlines()[0],
        "c_flags": cfg["c"]["flags"],
        "threads": "all kernels single-threaded; no numerical library inside the timed code",
        "power_plugged": battery.power_plugged if battery else None,
    }
    import matplotlib
    import numpy
    import scipy
    env["numpy"] = numpy.__version__
    env["scipy"] = scipy.__version__
    env["matplotlib"] = matplotlib.__version__
    if sys.platform == "win32":
        env["cpu_name"] = powershell("(Get-CimInstance Win32_Processor).Name")
        env["cpu_max_clock_mhz"] = powershell("(Get-CimInstance Win32_Processor).MaxClockSpeed")
        env["l2_cache_kb"] = powershell("(Get-CimInstance Win32_Processor).L2CacheSize")
        env["l3_cache_kb"] = powershell("(Get-CimInstance Win32_Processor).L3CacheSize")
        env["computer_model"] = powershell("(Get-CimInstance Win32_ComputerSystem).Model")
        env["power_plan"] = cmd_output(["powercfg", "/getactivescheme"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(env, f, indent=2)
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
