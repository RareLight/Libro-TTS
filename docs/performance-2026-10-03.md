# Resource measurements — October 3, 2026

Measured with project Python 3.12.2, MLX 0.30.6, and the locked mlx-audio revision `e42e1431fcf89af313375296c46d03a0153c4aa7` on macOS arm64. Machine-readable results are in [performance-2026-10-03.json](performance-2026-10-03.json). These are local measurements, not cross-machine performance guarantees.

## Method

The benchmark uses the application's serial generator or spawned pool, model loading, ordered full-waveform IPC, and atomic WAV output. It includes preparation, pool startup, generation, and encoding in each runtime wall time. A fresh child runs each mode. Serial repeats reuse the process's loaded model; parallel repeats create new pools and reload worker models. The first run is a first-process measurement, not a cleared disk-cache or first-download measurement. Controller startup/preflight are reported separately.

Real runs used existing Kokoro/Soprano snapshots in a disposable copied model tree. Internet socket connections were blocked in both the controller's generation child and synthesis workers. No new weights were downloaded. The three-chunk tests used the beginning of `input.txt`; the 40-chunk Kokoro run covered all of that file's 40 chunks. Real runs retained catalog settings and full precision.

RSS was sampled about every 0.2 seconds using the generation process and descendants. It can miss short peaks and double-count shared pages. The JSON also records the generation parent's process high-water RSS; that value excludes workers and accumulates across repeats. Incremental PCM writes count toward generation time; its final encoding time is header finalization and atomic publication. Synthetic fixtures measure PCM/output overhead, not speech generation speed.

## Results

| Input/output path | Workers | First / repeat runtime (seconds) | Sampled process-tree peak (MiB) |
|---|---:|---:|---:|
| 100 fixture chunks, buffered WAV | 1 | 1.034 / 0.808 | 1,569.8 |
| 100 fixture chunks, incremental WAV | 1 | 0.426 / 0.073 | 79.1 |
| 100 fixture chunks, buffered WAV | 2 | 1.314 / 1.201 | 1,543.4 |
| 100 fixture chunks, incremental WAV | 2 | 0.615 / 0.395 | 311.5 |
| Kokoro, 3 chunks, buffered WAV | 1 | 3.732 / 1.037 | 634.2 |
| Kokoro, 3 chunks, buffered WAV | 2 | 2.904 / 2.615 | 1,335.3 |
| Kokoro, 3 chunks, incremental WAV | 1 | 4.117 / 1.202 | 615.0 |
| Kokoro, 3 chunks, incremental WAV | 2 | 3.196 / 2.885 | 1,358.5 |
| Soprano, 3 chunks, incremental WAV | 1 | 4.985 / 1.317 | 430.2 |
| Soprano, 3 chunks, incremental WAV | 2 | 3.989 / 2.392 | 946.8 |
| Kokoro, 40 chunks, incremental WAV | 1 | 21.011 / not repeated | 635.4 |
| Kokoro, 40 chunks, incremental WAV | 2 | 18.833 / not repeated | 1,364.2 |

All recorded runs succeeded. The fixture produces 1,000 seconds of audio from 96,000,000 bytes of float32 PCM. Its buffered path retains both chunks and a combined array, then the pinned encoder builds a full-file Python sample list. Incremental output eliminates the combined array and that sample list. The serial incremental parent's high-water RSS was 91.2 MiB, confirming the improvement independently of the lower sampled figure.

Kokoro produced 25.925 seconds for each three-chunk run and 367.825 seconds for each 40-chunk mode (mono, 24 kHz). The long run's real-time factors were 0.0571 serial and 0.0512 parallel. Soprano produced 19.328 seconds serial and 18.112 / 19.136 seconds parallel (mono, 32 kHz); its stochastic durations prevent claims of identical speech or speed-normalized quality. No listening assessment was performed.

## Decisions

- Incremental WAV is enabled by default. It emits the same PCM16 samples as the pinned writer for verified mono/stereo numeric cases, keeps chunk order, and preserves the atomic replacement boundary. Parallel work still retains at most one configured batch of chunk results. A full serial retry truncates staged parallel audio before writing again. MP3/FLAC and other formats retain the existing buffered encoder.
- Keep the existing two-worker default, three-chunk threshold, 12-chunk batch, and CSM/Dia serial guards. Short warm serial runs beat repeated parallel startup, while the 40-chunk Kokoro run modestly favored parallel at higher memory cost. Use `--serial` on constrained machines or for short batch inputs; `--workers` and `--parallel-batch-size` allow explicit tuning.
- Retain per-file pools for now. Reusing them across batch files would change model/reference/RNG lifetime and failure cleanup. This limited evidence does not establish that the added lifecycle complexity is worthwhile. Parent model loading is already avoided on successful parallel work; batch transcript preparation is reused independently.
- Large-family resource measurements, hours-long real speech, remaining-family speech acceptance, and listening remain open. The deterministic 1,000-second fixture does not replace those gates. Whisper/Chatterbox verification and relocated releases remain excluded/deferred as requested.

## Reproduction

```bash
# No models or Metal required; compare the output-memory paths.
.venv/bin/python scripts/benchmark_chunk_parallel.py --fixture --offline \
  --max-chunks 100 --fixture-frames 240000 --repeats 2 --json-out tests_output/fixture-stream.json
.venv/bin/python scripts/benchmark_chunk_parallel.py --fixture --offline \
  --max-chunks 100 --fixture-frames 240000 --repeats 2 --buffered-wav --json-out tests_output/fixture-buffered.json

# Existing local models; fails rather than downloading if assets are missing.
.venv/bin/python scripts/benchmark_chunk_parallel.py --input-file input.txt \
  --model kokoro --offline --max-chunks 40 --repeats 1 --json-out tests_output/kokoro-resource.json
```

Use `--models-dir` to select a disposable model root. `--max-chunks` deliberately limits the input, so record whether a result covers the whole file. A failed parallel benchmark is reported as a failure rather than silently timed as serial fallback. Each mode has a bounded timeout and cleans up its owned process group.
