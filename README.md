# RFonEdge
The code in this repository implements the algorithms and experiments in the following paper:
> T. Jian, Y. Gong, Z. Zhan, R. Shi, N. Soltani, Z. Wang, J. Dy, K. Chowdhury, Y. Wang, S. Ioannidis, “Radio Frequency Fingerprinting on the Edge”, IEEE Transactions on Mobile Computing, 2021.

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
We implement progressive model pruning algorithm that progressively prune the pre-trained model that satisfies the pre-defined sparsity constraint sets for filter or column pruning. 

We evaluate our algorithm on five benchmark datasets, including three WiFi datasets (WiFi-50, WiFi-Eq-50, WiFi-Eq-500), one ADS-B dataset (ADS-B-50), and one mixture dataset (Mixture-50) containing both WiFi and ADS-B transmissions. 

We run all algorithms via `main.py`, and provide several useful tools to define/check sparsity settings as follows:

- `testers.py` for quick checking of the sparsity.

- `flops.py` for quick checking of model FLOPS.

- `profile/config*.yaml` template the configuration files. Each represents a resulting pruning ratio.

- `run.sh` templates an example script for running the code.

# Citing This Paper
Please cite the following paper if you intend to use this code for your research.
> T. Jian, Y. Gong, Z. Zhan, R. Shi, N. Soltani, Z. Wang, J. Dy, K. Chowdhury, Y. Wang, S. Ioannidis, “Radio Frequency Fingerprinting on the Edge”, IEEE Transactions on Mobile Computing, 2021.

# Acknowledgements
Our work is supported by National Science Foundation under grants CCF-1937500 and CNS-1923789.
