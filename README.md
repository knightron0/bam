# Norm Balancing Optimizers
[Sarthak Mangla](https://www.sarthakmangla.com), [Abel Gurung](https://abelgurung.github.io)

This repository contains the code accompanying the blog post [Norm Balancing Optimizers](https://www.sarthakmangla.com/blog/bam). It implements BAM (Balanced Axis Momentum), a stripped-down Muon variant that replaces Newton–Schulz orthogonalization with SinkNorm.

The nanoGPT training scripts (with different optimizers) live in [nanogpt](https://github.com/knightron0/bam/tree/main/src/nanogpt). The CIFAR-10 MLP and ResNet-18 experiments can be run via [run.py](https://github.com/knightron0/bam/blob/main/src/run.py) using the configs in [config](https://github.com/knightron0/bam/tree/main/configs). We'll be updating this repository with `sbatch` scripts that we used to run our sweeps soon!

Note: this code was ported over from an experimental, private repo. If there are issues or broken scripts, let us know!
