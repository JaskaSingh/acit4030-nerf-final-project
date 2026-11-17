# Improving a Simplified Neural Radiance Field for Synthetic and Real Scenes

ACIT4030 Machine Learning for 3D Computer Vision, OsloMet, autumn 2026.
Final project by Asad Mohammad Awais (S374979) and Jaskamal Singh Thiara (S374983).

This repository holds the code and results for our final project report. We improve the simplified NeRF from Assignment 2 in stages on the course Lego and poster datasets, using PyTorch3D and PyTorch.

## Contents

- `ACIT4030_final_project.ipynb`: all code. It defines and trains every run in the report, evaluates them and makes the report figures. The saved outputs are from the last session, which only made figures.
- `results_table.md`: test PSNR, SSIM, LPIPS, iterations, training time and GPU for every run, and the mean and standard deviation over seeds. Seed repeats are named `<run>_s2` and `<run>_s3`, and `lego400` is Lego at 400 x 400 pixels.

The per-run result files, test renders, checkpoints and the course data are not included because of their size.

## Runs

| Run | In the report |
| --- | --- |
| `baseline_rerun` | Baseline (3k) |
| `baseline_long` | Baseline (20k) |
| `stage1` | Stage 1 |
| `stage1_256pts` | Stage 1 with 256 samples (control for Stage 2) |
| `stage2` | Stage 2 |
| `stage3` | Stage 3 (poster) |
| `stage4` | Stage 4a (poster) |
| `stage5` | Stage 5 |
| `stage5_short` | Stage 5, 5000 iterations (Lego) |
| `stage5_weak` | Stage 5 with distortion weight 0.001 (poster) |
| `final` | Stage 4a, 50,000 iterations (poster) |
| `kplanes` | K-Planes |
| `kplanes_256`, `kplanes_128` | K-Planes with planes up to 256 or 128 (Lego) |
| `kplanes_256_tv` | K-Planes up to 256 with ten times stronger TV (Lego) |
| `kplanes_fullcoarse` | K-Planes with a full-size coarse grid (Lego) |
| `kplanes_long` | K-Planes, 50,000 iterations (poster) |

## Running the notebook

1. The notebook runs in Google Colab with Google Drive. Put the course data `acit4030-data.zip` in `MyDrive/ACIT4030/`.
2. Choose an L4 GPU with High-RAM. The first cell installs PyTorch3D. If no build for the current PyTorch version is cached in Drive, it builds one, which takes about 10 minutes the first time.
3. TF32 is switched off in the first cell and must stay off. With TF32 on, the models trained to only about 13 dB on Lego on the L4.
4. In section 10, choose `DATASET` and the runs to start (`RUN_MAIN`, `RUN_EXTRA`, `RUN_LATER`). Several copies of the notebook can share one dataset's queue through `WORKER` and `N_WORKERS`. Each run saves checkpoints to Drive and continues from the last one after a disconnect.
5. With `MAKE_FIGURES = True`, section 12 makes the report figures from the finished runs.

Results are written to `MyDrive/ACIT4030/final_project/results/`. Outside Colab, set the `DATA_ROOT` and `RESULTS_ROOT` environment variables instead.

Most runs were trained on an L4 GPU. The 50,000-iteration runs and some seed repeats were trained on an A100 (see `results_table.md`). Figure 9 in the report is the training-time figure from the notebook with the point labels moved so they do not overlap.
