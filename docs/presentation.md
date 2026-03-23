---
marp: true
theme: default
paginate: true
math: mathjax
style: |
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  :root {
    --heading: #1a1a2e;
    --body: #2d2d3d;
    --muted: #6b7280;
    --accent: #2563eb;
    --border: #e2e5ea;
    --bg: #ffffff;
    --bg-off: #f8f9fb;
  }

  section {
    background: var(--bg);
    color: var(--body);
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 24px;
    padding: 48px 56px;
    line-height: 1.5;
  }

  h1 {
    color: var(--heading);
    font-weight: 700;
    font-size: 1.6em;
    margin-bottom: 0.5em;
    letter-spacing: -0.02em;
  }

  h2 {
    color: var(--heading);
    font-weight: 600;
    font-size: 1.1em;
    margin-top: 0;
    margin-bottom: 0.4em;
  }

  h3 {
    color: var(--muted);
    font-weight: 500;
    font-size: 0.8em;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.2em;
  }

  strong {
    font-weight: 600;
    color: #222233;
  }

  code {
    font-family: 'JetBrains Mono', 'SF Mono', monospace;
    background: var(--bg-off);
    border: 1px solid var(--border);
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.82em;
    color: var(--accent);
  }

  table {
    font-size: 0.76em;
    border-collapse: collapse;
    width: 100%;
    margin: 0.4em 0;
  }

  th {
    background: var(--bg-off);
    color: var(--heading);
    font-weight: 600;
    padding: 8px 14px;
    text-align: left;
    border-bottom: 2px solid var(--accent);
  }

  td {
    padding: 6px 14px;
    border-bottom: 1px solid var(--border);
  }

  img {
    border-radius: 4px;
    border: 1px solid var(--border);
  }

  ul, ol {
    padding-left: 1.3em;
  }

  li {
    margin-bottom: 0.2em;
  }

  li::marker {
    color: var(--accent);
  }

  section.title {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
  }

  section.title h1 {
    font-size: 2.1em;
    font-weight: 600;
    margin-bottom: 0.15em;
  }

  section.title h2 {
    font-weight: 400;
    color: var(--muted);
    font-size: 1.05em;
  }

  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    align-items: start;
  }

  .two-img {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-top: 0.3em;
  }

  .note {
    background: var(--bg-off);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 4px;
    padding: 10px 14px;
    margin-top: 0.4em;
    font-size: 0.85em;
  }

  .big-number {
    font-size: 4em;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.02em;
  }

  section::after {
    color: var(--muted);
    font-size: 0.55em;
  }

---

<!-- _class: title -->
<!-- _paginate: false -->

# CIFAR-10 Classification from Scratch

## CSI 4140/5140, Project 1

<br>

**Anton Sakhanovych & Gavin D'Hondt**, Group 5
March 2026

---

# Problem & Constraints

Classify 32x32 RGB images from CIFAR-10 into 10 categories (50k train, 10k test).

All components implemented from scratch. No `nn.Module`, no `torch.optim`.

| Category | Components |
|---|---|
| Layers | Conv2D (im2col), Fully Connected, ReLU, Sigmoid, Softmax, Flatten, Dropout |
| Optimizers | SGD, SGD + Momentum (EMA), Adam |
| LR Schedules | Cosine decay, Step decay |
| Regularization | L2 weight penalty, Inverted dropout |
| Data | Per-channel normalization, random flip, random crop |

---

# Model Architecture

Three conv blocks with **strided convolutions** for downsampling, then two FC layers.

<div class="two-col">
<div>

| Layer | Output |
|---|---|
| Conv(3 &rarr; C₁, s=1) + ReLU | (N, C₁, 32, 32) |
| Conv(C₁ &rarr; C₂, **s=2**) + ReLU | (N, C₂, 16, 16) |
| Conv(C₂ &rarr; C₃, **s=2**) + ReLU | (N, C₃, 8, 8) |
| Flatten | (C₃ &middot; 64, N) |
| FC &rarr; ReLU | (512, N) |
| FC &rarr; Softmax | (10, N) |

</div>
<div>

### Key details
- Convolutions use `F.unfold` (im2col): patches become columns, forward pass is a single matmul
- He initialization: $w \sim \mathcal{N}(0,\; 2/n_\text{in})$
- Strided convolutions let the network learn its own downsampling

### Baseline width
C₁=32, C₂=64, C₃=128

</div>
</div>

---

# Ablation Study

8 experiments, each varying **one factor** while holding the rest at baseline.

**Baseline:** Adam (lr=0.001, default betas), standard width, ReLU, augmentation on, no regularization. 15 epochs per experiment.

| Experiment | What was varied | Baseline result |
|---|---|---|
| Optimizer & LR | Adam vs SGD+Momentum | 78.15% |
| Adam betas | β₁ and β₂ | 76.61% |
| LR schedule | Cosine vs Step vs None | 77.53% |
| Data augmentation | On vs Off | 78.06% |
| Network width | Slim / Standard / Wide | 77.18% |
| Activation | ReLU vs Sigmoid | 77.78% |
| Regularization | None / L2 / Dropout / Both | 77.78% |
| L2 lambda | 10⁻⁴ through 10⁻¹ | 77.35% |

---

# Data Augmentation

The single largest factor in the ablation study: **+13 points** from random flip + crop (pad 4).

<div class="two-img">

![](images/exp6/exp6_augmentation_test_acc.png)

![](images/exp6/exp6_augmentation_loss.png)

</div>

---

# Data Augmentation (cont.)

|  | Test Acc | Train Acc | Train Loss |
|---|---|---|---|
| **Augmentation on** | **78.06%** | 78.21% | 0.6269 |
| Augmentation off | 64.86% | 98.62% | 0.0436 |

Without augmentation, training accuracy reaches 98.6% while test accuracy stalls at 64.9%. The model memorizes pixel patterns rather than learning generalizable features. Training loss collapses to near zero while test loss climbs past 2.0.

With augmentation on, training and test accuracy stay within a few points of each other throughout all 15 epochs.

---

# Optimizer: Adam vs SGD+Momentum

Adam (lr=0.001) outperforms SGD+Momentum at every learning rate tested.

<div class="two-img">

![](images/exp4/exp4_optimizer_test_acc.png)

![](images/exp4/exp4_optimizer_loss.png)

</div>

---

# Optimizer (cont.)

| Optimizer | LR | Test Acc | Train Loss |
|---|---|---|---|
| **Adam** | **0.001** | **78.15%** | **0.5969** |
| SGD+Momentum | 0.1 | 76.99% | 0.6548 |
| SGD+Momentum | 0.05 | 73.66% | 0.7752 |
| SGD+Momentum | 0.01 | 62.81% | 1.0870 |

Adam's per-parameter adaptive learning rates mean a single lr=0.001 works well. Momentum is very sensitive to the global learning rate: competitive at 0.1, but barely converges at 0.01.

---

# Activation: ReLU vs Sigmoid

Sigmoid falls over **10 points** behind ReLU due to vanishing gradients.

<div class="two-img">

![](images/exp8/exp8_activation_test_acc.png)

![](images/exp8/exp8_activation_loss.png)

</div>

---

# Activation (cont.)

| Activation | Test Acc | Train Loss |
|---|---|---|
| **ReLU** | **77.78%** | 0.6260 |
| Sigmoid | 67.23% | 0.9501 |

Sigmoid saturates at both extremes of its output range, driving the derivative toward zero. Across multiple layers, the gradient product shrinks exponentially, starving early layers of any training signal.

He initialization (which we use) is also designed specifically for ReLU. It accounts for the fact that ReLU zeroes out roughly half the neurons and scales the initial weights to compensate. That assumption does not hold for Sigmoid.

---

# L2 Regularization Strength

The usable range for L2 is narrow. One order of magnitude separates "no effect" from "training failure."

<div class="two-img">

![](images/exp3/exp3_l2_lambda_test_acc.png)

![](images/exp3/exp3_l2_lambda_loss.png)

</div>

---

# L2 Strength (cont.)

| λ | Test Acc | Observation |
|---|---|---|
| 10⁻⁴ | 77.35% | Near baseline |
| 10⁻³ | 73.80% | Mild slowdown |
| 10⁻² | 51.49% | Penalty dominates updates |
| 10⁻¹ | 23.74% | Near random chance |

Below 10⁻⁴, the penalty gradient is too small to compete with the data gradient. At 10⁻², the penalty has taken over the update step entirely, so weight updates are dominated by shrinkage rather than the loss signal.

---

# Regularization Method

No regularization wins at 15 epochs because the model has not started overfitting yet.

<div class="two-col">
<div>

![](images/exp2/exp2_regularization_test_acc.png)

</div>
<div>

| Method | Test Acc | Train Acc |
|---|---|---|
| **None** | **77.78%** | 77.88% |
| Dropout (p=0.3) | 74.23% | 70.70% |
| L2 (λ=0.001) | 73.01% | 70.79% |
| L2 + Dropout | 71.65% | 66.36% |

Training and test accuracy are within 1 point of each other. There is no overfitting to prevent, so regularization only slows convergence.

</div>
</div>

---

# Colab Training Challenges

Several issues had to be resolved to train on Google Colab's free GPU tier.

| Problem | Cause | Fix |
|---|---|---|
| ~14 GB memory from autograd | `nn.Parameter` defaults to `requires_grad=True`, so PyTorch builds a computation graph even though we compute gradients manually | Wrapped forward/backward in `torch.no_grad()` |
| OOM in Adam/Momentum | Each optimizer step allocated ~6 intermediate tensors per parameter | Switched to in-place ops (`mul_`, `add_`, `addcmul_`) |
| OOM from cached activations | Forward pass caches (unfolded input, ReLU mask) persisted through the full backward pass | Freed caches immediately after each layer's backward |
| Device mismatch errors | Parameters created on CPU, training data moved to GPU | Added lazy `.to(device)` for params and optimizer state |
| NaN loss | `log(0)` when softmax output is near zero | Clamped probabilities to a minimum of 1e-12 |

---

# Final Configuration

Best setting from each ablation experiment, trained for 50 epochs.

<div class="two-col">
<div>

| Component | Choice |
|---|---|
| Optimizer | Adam, lr=0.001 |
| Betas | β₁=**0.8**, β₂=**0.99** |
| Width | **Wide** (64, 128, 256) |
| LR Schedule | Step decay, γ=0.5 / 5 ep |
| Augmentation | Flip + crop (pad 4) |
| Regularization | None |
| Epochs | 50 |

</div>
<div>

### Changes from baseline
- β₁ lowered from 0.9 to 0.8 (+1 pt)
- β₂ lowered from 0.999 to 0.99
- Filters doubled at each stage (+1.7 pt)
- Step decay for late-stage refinement
- Training extended from 15 to 50 epochs

</div>
</div>

---

# Final Results

<div class="two-img">

![](images/final/final_test_acc.png)

![](images/final/final_train_acc.png)

</div>

---

# Final Results (cont.)

<div class="two-col">
<div>

![](images/final/final_loss.png)

</div>
<div>

- Crosses **80%** at epoch 13
- Peaks at **82.51%** at epoch 48
- After epoch 30, accuracy settles in a narrow band above 82%
- Clears the 75% requirement and the 80% extra credit threshold
- **+4.36 points** over the ablation baseline of 78.15%

</div>
</div>

---

<!-- _class: title -->
<!-- _paginate: false -->

<div class="big-number">82.51%</div>

## Final Test Accuracy

Wider filters · Tuned Adam betas · Step decay · 50 epochs

<br>

Anton Sakhanovych & Gavin D'Hondt, Group 5
