# RFonEdge

This is the released repo for our work entitled `Radio Frequency Fingerprinting on the Edge`. 

## Prerequisites
```bash
pytorch-1.6.0
torchvision-0.7.0
numpy-1.16.1
scipy-1.3.1
tqdm-4.33.0
yaml-0.1.7
```

## Experiments
main.py for main program.

testers.py for quick checking of the sparsity.

flops.py for quick checking of model FLOPS.

profile/config*.yaml template the configuration files. Each represents a resulting pruning ratio.

run.sh templates an example script for running the code.
