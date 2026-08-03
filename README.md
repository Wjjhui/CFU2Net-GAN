# CFU2Net-GAN Method-aligned implementation

This package contains the revised project files, the consistency audit, a smoke test, the evaluation implementation, and the day-level statistical-analysis utility.

## Installation

```bash
pip install -r requirements.txt
```

## Static and smoke checks

```bash
python -m compileall .
python smoke_test.py
```

## Training the complete CFU2Net-GAN

```bash
python Trainer.py \
  --train-dir /path/to/2019_2021_train \
  --val-dir /path/to/2022_validation \
  --output-dir output/cfu2net_gan \
  --model-variant complete \
  --epochs 200 \
  --batch-size 4 \
  --learning-rate 0.001 \
  --warmup-steps 1000 \
  --patience 10 \
  --seed 0
```

## Resuming training

```bash
python trainBreakPoint.py \
  --train-dir /path/to/2019_2021_train \
  --val-dir /path/to/2022_validation \
  --output-dir output/cfu2net_gan \
  --model-variant complete \
  --resume output/cfu2net_gan/checkpoints/latest.pth
```

## Evaluating the 2023 test set

```bash
python assess.py \
  --checkpoint output/cfu2net_gan/best_validation_gen.pth \
  --data-dir /path/to/2023_test \
  --model-variant complete \
  --threshold 0.20 \
  --radius 1 \
  --output-csv evaluation/cfu2net_gan.csv
```

## Day-level paired statistics

```bash
python statistical_analysis.py \
  --reference evaluation/cfu2net_gan.csv \
  --comparison ConvLSTM=evaluation/convlstm.csv \
  --comparison TU2Net-GAN=evaluation/tu2net_gan.csv \
  --comparison DGMR=evaluation/dgmr.csv \
  --bootstrap-resamples 10000 \
  --output evaluation/day_level_statistics.csv
```

To reproduce the manuscript-wide 48-comparison Benjamini-Hochberg family, provide all eight baseline CSV files in the same command.

## Generating prediction figures

```bash
python sampleCloud.py \
  --checkpoint output/cfu2net_gan/best_validation_gen.pth \
  --data-dir /path/to/2023_case_samples \
  --output-dir output/cases \
  --model-variant complete
```

The exact unresolved items and the complete Method-to-code mapping are documented in `ALIGNMENT_REPORT.md`.
