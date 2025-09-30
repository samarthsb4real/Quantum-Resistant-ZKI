# Hash Function Comparative Analysis Report
Generated on: 2025-03-24 15:42:24

## Sample Hash Outputs
**SHA-512**: `849d1697c7a48b7413a131e20d3665ced5b2410bf234eb18da3aa66dc82e1c98bf981f0224004187a1fbcb851412baacf421aff58f01ae0c40a2f7d1f104107a`

**Quantum-Safe Hash**: `c6376b48a4628a71d48d4eb95aa5c1a597002e8d6fc37d2f7bb3263882524a12`

## Performance Analysis
Average execution time in seconds:

```
+-------------------+------------+--------------+---------------+
| Hash Function     |   64 bytes |   1024 bytes |   16384 bytes |
+===================+============+==============+===============+
| SHA-512           | 6.603e-07  |   1.9956e-06 |   1.77819e-05 |
+-------------------+------------+--------------+---------------+
| Quantum-Safe Hash | 1.5553e-06 |   3.6114e-06 |   2.25542e-05 |
+-------------------+------------+--------------+---------------+
```

## Entropy Analysis
Score closer to 1.0 indicates better randomness distribution:

```
+-------------------+-----------------+
| Hash Function     |   Entropy Score |
+===================+=================+
| SHA-512           |               1 |
+-------------------+-----------------+
| Quantum-Safe Hash |               1 |
+-------------------+-----------------+
```

## Collision Resistance
```
+-------------------+--------------+--------+----------+
| Hash Function     |   Collisions |   Rate | Status   |
+===================+==============+========+==========+
| SHA-512           |            0 |      0 | PASSED   |
+-------------------+--------------+--------+----------+
| Quantum-Safe Hash |            0 |      0 | PASSED   |
+-------------------+--------------+--------+----------+
```

## Memory Usage
```
+-------------------+----------------+
| Hash Function     | Memory Usage   |
+===================+================+
| SHA-512           | 153 bytes      |
+-------------------+----------------+
| Quantum-Safe Hash | 89 bytes       |
+-------------------+----------------+
```

## Avalanche Effect
Ideal score is 50% bit changes (lower ideal_score is better):

```
+-------------------+----------------+---------------+
| Hash Function     | Bit Change %   |   Ideal Score |
+===================+================+===============+
| SHA-512           | 49.98%         |          0.02 |
+-------------------+----------------+---------------+
| Quantum-Safe Hash | 49.91%         |          0.09 |
+-------------------+----------------+---------------+
```

## Summary and Recommendations
- **Fastest Performance**: SHA-512
- **Best Entropy**: SHA-512
- **Best Avalanche Effect**: SHA-512

### Overall Recommendation
Based on the comprehensive analysis above, the recommended hash function depends on your specific requirements:
- For performance-critical applications: Consider using SHA-512
quantum_safe_hash
- For general-purpose cryptographic applications: Balance between performance and security based on your specific needs