# Limitations and Validity Notes

## Key Limitations
1. **Quantum attack execution is modeled**
   - Grover/BHT outputs are based on query complexity with calibrated classical throughput.
   - They are not results from quantum circuits executed on quantum hardware.

2. **Hash-level focus**
   - Results address hash construction properties and benchmark behavior.
   - End-to-end protocol, key management, and transport-layer guarantees are out of scope.

3. **Environment sensitivity**
   - Throughput and latency depend on CPU, thermal state, OS scheduling, and Python runtime.

4. **Empirical bounds**
   - Collision/preimage experiments are bounded by finite samples and budgets.

## Internal Validity Controls
- Fixed benchmark protocol and repeated measurements
- Deterministic mode + randomized mode
- Environment metadata capture in every run
- Baseline comparisons against standard hashes

## External Validity Guidance
- Re-run on multiple hardware profiles
- Add ARM/x86 cross-platform runs
- Add CI trend tracking for drift detection

## Recommended Paper Language
Use statements like:
- "Modeled Grover/BHT complexity suggests..."
- "Empirical tests did not observe collisions within the tested budget..."
- "Results are reproducible under the documented runtime configuration..."
