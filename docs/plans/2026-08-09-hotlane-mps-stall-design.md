# Hotlane MPS Stall Recovery Design

**Problem:** The hotlane reserves embedding slices but completes zero vectors because lazy model initialization selects MPS and blocks indefinitely while transferring model weights to Metal.

## Evidence

The live PID reached `_run_split_cycle()` and closed its read-only database connection before model initialization. A native sample then placed every main-thread observation in this chain:

`SentenceTransformer(..., device="mps")` → `torch.nn.Module.to` → `mps_copy_` → `MPSStream::synchronize` → `-[MTLCommandBuffer waitUntilCompleted]`.

The write path follows embedding, so a blocked device transfer prevents `_write_embedded_vectors()` from opening a connection or starting a vector transaction.

## Options

1. **Make hotlane CPU-only by default (selected).** Extend the shared embedding wrapper with an explicit device override, then have the hotlane request `cpu`. This isolates the reliability trade-off to a four-item background batch while preserving automatic MPS selection for all existing consumers.
2. **Wrap MPS initialization in a Python timeout.** Rejected because the block is inside a native Metal synchronization call; a timer or worker thread cannot safely interrupt and recover the wedged process.
3. **Run embedding in a supervised child process with MPS-to-CPU fallback.** Technically robust, but it adds IPC, lifecycle management, and expensive model reload behavior that is unnecessary for the current batch-four throughput requirement.

## Design

`EmbeddingModel` and `get_embedding_model()` accept an optional device override. `None` retains today's automatic `mps`/`cpu` choice; an explicit value is forwarded directly to `SentenceTransformer`. The singleton key includes the device so an explicit CPU wrapper cannot reuse an automatically selected MPS wrapper.

The hotlane's `run()` requests `cpu` from device-aware factories. Test factories that intentionally expose only a zero-argument call remain supported through the existing signature-inspection pattern. The daemon does not attempt MPS first: avoiding the unrecoverable native wait is the safety property.

## Verification

- RED/GREEN unit test that the hotlane asks a device-aware model factory for `cpu` before any cycle.
- Unit tests that explicit device selection bypasses automatic MPS probing and that the global wrapper cache distinguishes explicit CPU from automatic selection.
- Existing hotlane and lazy-startup tests.
- Full project test and lint gates.
- After merge, kickstart the live LaunchAgent from the merged checkout and record a strictly increasing vector count for 15 consecutive one-minute samples.
