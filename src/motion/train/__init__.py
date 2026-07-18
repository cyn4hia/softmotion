"""GPU fine-tuning entrypoints for SAM 3 and LocateAnything, plus GGUF conversion
so a fine-tuned LocateAnything runs back on the Mac via ggml. All heavy imports are
lazy: importing this package on a CPU machine is safe."""
