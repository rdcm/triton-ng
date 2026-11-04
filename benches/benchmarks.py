import argparse
import csv
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import perf_analyzer

PERF_ANALYZER = Path(perf_analyzer.__file__).parent / "bin" / "perf_analyzer"

BACKENDS = {"rs": "triton-ng", "py": "python_backend"}


@dataclass(frozen=True)
class Benchmark:
    name: str
    shape: str | None = None
    streaming: bool = False


BENCHMARKS = [
    Benchmark("argmax"),
    Benchmark("infer"),
    Benchmark("infer_concurrent"),
    Benchmark("decoupled", shape="Input3:4,1,28,28", streaming=True),
    Benchmark("stream", shape="Input3:4,1,28,28"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="python_backend vs triton-ng with perf_analyzer")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--http-port", type=int, default=9000)
    parser.add_argument("--grpc-port", type=int, default=9001)
    parser.add_argument("--concurrency", default="1,4,16,64")
    parser.add_argument("--measurement-interval", type=int, default=5000)
    parser.add_argument("--only", help="comma separated benchmark names")
    parser.add_argument("--results", default="../target/benchmarks")
    args = parser.parse_args()
    args.concurrency = [int(c) for c in args.concurrency.split(",")]
    args.benchmarks = [
        b for b in BENCHMARKS if args.only is None or b.name in args.only.split(",")
    ]
    return args


def http_get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode()


def wait_for_models(args, models):
    deadline = time.monotonic() + 300
    while True:
        try:
            if all(ready(args, model) for model in models):
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        if time.monotonic() > deadline:
            sys.exit(f"models {models} are not ready after 300 s")
        time.sleep(1)


def ready(args, model):
    url = f"http://{args.host}:{args.http_port}/v2/models/{model}/ready"
    try:
        http_get(url)
        return True
    except urllib.error.HTTPError:
        return False


def run(args, benchmark, model, concurrency):
    report = Path(args.results) / f"{model}_c{concurrency}.csv"
    command = [
        str(PERF_ANALYZER),
        "-m", model,
        "-u", f"{args.host}:{args.grpc_port}",
        "-i", "grpc",
        "--async",
        "--concurrency-range", str(concurrency),
        "--measurement-interval", str(args.measurement_interval),
        "-f", str(report),
    ]
    if benchmark.shape:
        command += ["--shape", benchmark.shape]
    if benchmark.streaming:
        command += ["--streaming"]
    result = subprocess.run(command, capture_output=True, text=True)
    (Path(args.results) / f"{model}_c{concurrency}.log").write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        sys.exit(f"{' '.join(command)}\n{result.stdout}{result.stderr}")
    with report.open() as f:
        row = next(csv.DictReader(f))
    return {
        "throughput": float(row["Inferences/Second"]),
        "p50_ms": int(row["p50 latency"]) / 1000,
        "p99_ms": int(row["p99 latency"]) / 1000,
    }


def table(benchmark, rows):
    lines = [
        f"### `{benchmark.name}`",
        "",
        "| Concurrency | triton-ng, infer/s | python_backend, infer/s | Speedup | triton-ng p50 / p99, ms | python_backend p50 / p99, ms |",
        "|---|---|---|---|---|---|",
    ]
    for concurrency, rs, py in rows:
        lines.append(
            f"| {concurrency} | {rs['throughput']:.0f} | {py['throughput']:.0f} "
            f"| {rs['throughput'] / py['throughput']:.2f}× "
            f"| {rs['p50_ms']:.2f} / {rs['p99_ms']:.2f} | {py['p50_ms']:.2f} / {py['p99_ms']:.2f} |"
        )
    return "\n".join(lines)


def environment(args):
    cpu = "unknown"
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    memory_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    server = json.loads(http_get(f"http://{args.host}:{args.http_port}/v2"))
    version = subprocess.run([str(PERF_ANALYZER), "--version"], capture_output=True, text=True)
    gpu = "none"
    if shutil.which("nvidia-smi"):
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        gpu = smi.stdout.strip() or "none"
    return "\n".join(
        [
            f"- CPU: {cpu}, {os.cpu_count()} threads",
            f"- RAM: {memory_gb:.0f} GiB",
            f"- GPU: {gpu} (not used by the benchmark models)",
            f"- OS: {platform.platform()}",
            f"- Triton: {server['version']} (nvcr.io/nvidia/tritonserver:25.09-py3)",
            f"- perf_analyzer: perf-analyzer {importlib.metadata.version('perf-analyzer')} from PyPI "
            f"(the binary reports {version.stdout.strip() or version.stderr.strip()})",
            f"- Concurrency: {', '.join(map(str, args.concurrency))}; "
            f"measurement window {args.measurement_interval} ms, stability 10 %",
        ]
    )


def main():
    args = parse_args()
    Path(args.results).mkdir(parents=True, exist_ok=True)
    models = [f"{b.name}_{suffix}" for b in args.benchmarks for suffix in BACKENDS]
    wait_for_models(args, models)

    sections = ["## Environment", "", environment(args), ""]
    print("\n".join(sections), flush=True)
    for benchmark in args.benchmarks:
        rows = []
        for concurrency in args.concurrency:
            rs = run(args, benchmark, f"{benchmark.name}_rs", concurrency)
            py = run(args, benchmark, f"{benchmark.name}_py", concurrency)
            rows.append((concurrency, rs, py))
        section = table(benchmark, rows)
        print(section + "\n", flush=True)
        sections += [section, ""]
    (Path(args.results) / "results.md").write_text("\n".join(sections))


if __name__ == "__main__":
    main()
