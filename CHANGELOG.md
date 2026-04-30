# Changelog

All notable changes to LAKER-XSA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive benchmark suite (10 benchmarks complete)
- Long sequence scaling analysis (128, 256 tokens)
- NLP sentiment classification benchmark
- Documentation: RESULTS.md, FINAL_SUMMARY.md, BENCHMARK_STATUS.md

### Changed
- Updated documentation with complete benchmark results

### Fixed
- Dimension mismatch in SentimentClassifier (removed vocab output projection)

## [0.1.0] - 2026-04-30

### Added
- Core implementation in `src/laker_xsa/`
  - `config.py` - Configuration dataclass
  - `attention/` - Standard, XSA, Kernel, and Fused attention
  - `solver/` - Preconditioned Richardson iteration, Conjugate Gradient
  - `model/` - Transformer block and full model
  - `training/` - Training utilities
  - `benchmarks/` - Performance benchmarking tools
  - `utils/` - Helper utilities
- Test suite in `tests/`
  - Shape verification tests
  - Gradient flow tests
  - Numerical stability tests
- Example scripts in `examples/`
  - `comparative_analysis.py` - Easy synthetic tasks
  - `hard_benchmark.py` - Challenging algorithmic tasks
  - `long_sequence_benchmark.py` - Scaling analysis
  - `nlp_sentiment_benchmark.py` - NLP evaluation
- Documentation in `docs/`
  - `architecture.md` - System overview
  - `math.md` - Mathematical derivations
  - `design_decisions.md` - Implementation choices
  - `limitations.md` - Known limitations
  - `benchmark_report.md` - Benchmark methodology
  - `QUANTITATIVE_SUMMARY.md` - Quantitative analysis
  - `BENCHMARK_STATUS.md` - Running status
  - `FINAL_SUMMARY.md` - Comprehensive summary
- MIT License
- pyproject.toml for package configuration
- README.md with quick start guide
