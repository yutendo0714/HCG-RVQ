# Experiment Progress

Date: 2026-05-29

## E000: Project Bootstrap

Status: Done

Notes:

- Created a clean top-level research project structure instead of keeping all work under `proposed/`.
- Preserved the pre-implementation literature and implementation reading notes in `docs/research_reading_notes.md`.
- Initial implementation target is a CompressAI-style MeanScaleHyperprior backbone with RVQ/HCS/HCG variants.

Next:

- Install dependencies into `.venv`.
- Run a CPU smoke test on a random tensor.
- Start scalar baseline and tiny RVQ pilot configs.


## E001: Environment and Smoke Test

Status: Done

Results:

- Created `.venv` and installed project dependencies.
- Corrected PyTorch to `2.7.1+cu128` so CUDA works with the local CUDA 12.8 driver.
- Verified imports: torch, CompressAI, wandb, numpy.
- Verified CUDA visibility on NVIDIA GeForce RTX 3090.
- Ran random forward/loss checks for scalar, global RVQ, HCS-RVQ, and HCG-RVQ-H variants.
- Ran CPU tiny smoke training on Kodak for 1 epoch and saved `experiments/tiny_smoke/checkpoint_latest.pth.tar`.
- Ran CPU eval smoke on 2 Kodak images: bpp 0.178448, PSNR 9.322829, MS-SSIM 0.294151.

Next:

- Run the first real scalar baseline with wandb enabled.
- Then run `global_rvq`, `hcs_rvq`, and `hcg_rvq_h` under the same tiny/pilot schedule.
