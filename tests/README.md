# 🧪 Verification & Benchmark Test Suite

This directory contains integration test suites and stress benchmarks for verifying RakshaVision before deployment.

## Test Scripts
- `test_pipeline.py`: Comprehensive 5-stage automated integration test verifying detector loading, 4-point PPE evaluation, hazard detection, CSV audit logging, and HUD visualization.
- `benchmark_harsh_conditions.py`: Stress-testing suite evaluating zero false-positive rates for normal shirts vs. vests, steam/dust rejection, and 15-iteration edge latency.
