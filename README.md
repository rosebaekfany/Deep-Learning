# Deep Learning

Homework assignments and the two-phase final project for the **Deep Learning** course (Electrical Engineering, Sharif University of Technology).

## Final Project — Multi-Object Tracking in Sports Videos

| Phase | File | Summary |
|---|---|---|
| 1 — Literature review | [`deep_project_phase1.pdf`](deep_project_phase1.pdf) | Review of multi-object tracking (MOT) methods — **SORT, DeepSORT, FairMOT, ByteTrack** — applied to tracking players, referees and the ball. Covers metrics (MOTA, IDF1, precision/recall, fragmentation), benchmarks (**SportsMOT, MOT17, DanceTrack, PoseTrack**) and open challenges such as occlusion, scale variation and illumination changes. |
| 2 — Implementation | [`deep-project_phase2_final_ZahraMaleki.ipynb`](deep-project_phase2_final_ZahraMaleki.ipynb) | **YOLO** detection combined with **ByteTrack / DeepSORT** tracking on **SportsMOT**, with analysis of failure cases (occlusion, scale, illumination) and improvements such as occlusion-aware re-detection. |

## Homeworks ([`myHws/`](myHws))

| Notebook | Topic |
|---|---|
| `knn.ipynb` | k-nearest neighbours and cross-validation |
| `SVM.ipynb` | Support vector machines |
| `CART.ipynb` | CART decision trees on MNIST |
| `hw2_1.ipynb` | Optimisation algorithms (gradient descent, Newton's method) and an MLP for MNIST |
| `DL-HW2-P2.ipynb` | Loss functions and multilayer perceptrons |
| `HW3_ZahraMaleki_400110009_p.ipynb` | Cross-entropy vs. triplet loss for CNN feature learning |
| `deeplearning-hw4-p1.ipynb` | Emotion recognition on tweets with recurrent neural networks |
| `DeepLearning_HW4_p2.ipynb` | Language modelling on WikiText-2 |
| `deeplearning-hw4-p3.ipynb` | Encoder–decoder and decoder-only language models |
| `Q1_VAE_CVAE.ipynb` | Variational and conditional variational autoencoders |
| `Q2_VQ_VAE.ipynb` | Vector-quantised VAEs (VQ-VAE) |

## From-scratch classifiers (repository root)

Implementations of the classic image-classification pipeline on CIFAR-10, written without deep-learning frameworks:

- `knn.ipynb`, `svm.ipynb`, `softmax.ipynb` — k-NN, multiclass SVM and softmax classifiers
- `two_layer_net.ipynb` — fully connected network with hand-written forward/backward passes
- `classifiers/`, `layers.py`, `layer_utils.py`, `optim.py`, `solver.py`, `gradient_check.py` — the supporting library code

## Tools

Python · NumPy · PyTorch · Jupyter
