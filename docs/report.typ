#set document(title: "CSI 4140/5140 — Project 1 Report")
#set page(margin: 1in)
#set text(size: 12pt)
#set heading(numbering: "1.")
#set par(justify: true)

#align(center)[
  #text(size: 16pt, weight: "bold")[CSI 4140/5140 — Project 1 Report]

  #v(0.5em)
  Group 5 \
  Anton Sakhanovych and Gavin D'Hondt

  #v(0.5em)
  #datetime.today().display("[month repr:long] [day], [year]")
]

= Problem Statement

The goal is to classify 32 #sym.times 32 RGB images from the CIFAR-10 dataset into 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The dataset has 50,000 training images and 10,000 test images.

The constraint is that all layers, optimizers, and schedulers must be implemented from scratch in PyTorch without using any built-in nn.Module layers or optimizer classes. This forces us to work through the math directly rather than calling into pre-built abstractions.

= Design and Implementation

== Model Architecture

The model is a three-block CNN followed by two fully connected layers. We use strided convolutions for spatial downsampling rather than pooling layers.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: 0.5pt,
    [*Layer*], [*Input Shape*], [*Output Shape*],
    [Conv(3 #sym.arrow.r 32, k=3, s=1, p=1) + ReLU], [(N, 3, 32, 32)],   [(N, 32, 32, 32)],
    [Conv(32 #sym.arrow.r 64, k=3, s=2, p=1) + ReLU], [(N, 32, 32, 32)], [(N, 64, 16, 16)],
    [Conv(64 #sym.arrow.r 128, k=3, s=2, p=1) + ReLU], [(N, 64, 16, 16)], [(N, 128, 8, 8)],
    [Flatten], [(N, 128, 8, 8)], [(8192, N)],
    [FC(8192 #sym.arrow.r 512) + ReLU], [(8192, N)], [(512, N)],
    [FC(512 #sym.arrow.r 10) + Softmax], [(512, N)], [(10, N)],
  ),
  caption: [Baseline model architecture. N is batch size. Conv layers use (N, C, H, W); fully connected layers use (features, N).]
)

Here is how each component works:

- *Convolutional layer:* A convolution is just a dot product between the kernel and each local patch of the input. Instead of looping over every position, we use `F.unfold` to collect all patches into a single matrix: for each of the $H_"out" dot W_"out"$ positions, it extracts the $C_"in" dot K^2$ values the kernel would see and lays them out as a column, giving shape $(N, C_"in" dot K^2, H_"out" dot W_"out")$. Then one matrix multiply with the weight matrix (reshaped to $(C_"out", C_"in" dot K^2)$) computes all output positions at once. For the backward pass, the weight gradient is $nabla_"mat" dot X_"unfolded"^top$ summed over the batch (each output position contributed to a weight based on what patch it saw), and the input gradient is $W^top dot nabla_"mat"$ which gives the per-patch error signals back in column form, then `F.fold` adds overlapping contributions together and reconstructs the spatial layout.

- *Activation layers:* ReLU is $(x + |x|) / 2$; the backward pass multiplies the incoming gradient by $(x > 0)$. Softmax subtracts the per-sample max before exponentiation for numerical stability, then normalizes column-wise.

- *Flatten layer:* Reshapes $(N, C, H, W)$ to $(C dot H dot W, N)$ to match the column-major convention expected by the fully connected layers. The input shape is cached for the backward pass.

- *Fully connected layer:* Forward pass computes $Z = W X + b$ and applies the activation. Backward pass: $nabla W = nabla Z dot X^top$, $nabla b = sum_"batch" nabla Z$, $nabla X = W^top dot nabla Z$.

- *Dropout layer:* Uses inverted dropout. During training, a binary mask is sampled with keep probability $(1 - p)$ and immediately scaled by $1 / (1 - p)$, so the expected value of the output matches the input. The same mask is applied in the backward pass. Dropout is skipped entirely during evaluation.

== Methods

*Loss function.* Cross-entropy over the softmax probabilities:
$cal(L) = -1/N sum_(i=1)^N log p_(y_i)$,
where $p_(y_i)$ is the predicted probability for the true class. The gradient is $-1/(N p_(y_i))$ at the true class index and zero everywhere else.

*Optimizers.* All three share a `BaseOptimizer` base class that handles parameter iteration and optional regularization:

- *SGD:* $w arrow.l w - eta nabla_w cal(L)$

- *SGD + Momentum (EMA style):* maintains a velocity per parameter, $v arrow.l beta v + (1 - beta) nabla_w cal(L)$, then $w arrow.l w - eta v$. The EMA formulation keeps $v$ in the same magnitude range as the gradient, unlike the standard momentum formulation which accumulates unboundedly.

- *Adam:* $m arrow.l beta_1 m + (1 - beta_1) g$, $v arrow.l beta_2 v + (1 - beta_2) g^2$, with bias correction $hat(m) = m / (1 - beta_1^t)$, $hat(v) = v / (1 - beta_2^t)$, then $w arrow.l w - eta hat(m) / (sqrt(hat(v)) + epsilon)$. Defaults: $beta_1 = 0.9$, $beta_2 = 0.999$, $epsilon = 10^(-8)$.

*Learning rate schedules.* Cosine decay:
$eta_t = eta_"min" + 1/2 (eta_0 - eta_"min")(1 + cos(pi t / T))$.
Step decay multiplies the current learning rate by $gamma$ every $k$ epochs, floored at $eta_"min"$.

*Regularization.* L2 adds $2 lambda w$ to the weight gradient during the optimizer step. It is only applied to weight parameters, not biases. The regularizer is a separate protocol class, decoupled from the optimizer, so it can be swapped without touching the optimizer code. Dropout is applied after each convolutional activation during training.

*Data pipeline.* Images are normalized per channel with the CIFAR-10 dataset statistics: mean $(0.4914, 0.4822, 0.4465)$, std $(0.2470, 0.2435, 0.2616)$. The training loader applies random horizontal flip and random crop with padding 4. Test loader has no augmentation and is deterministic.

*Weight initialization.* He initialization for both convolutional and fully connected weights: $w tilde cal(N)(0, 2/n_"in")$, where $n_"in"$ accounts for kernel area in conv layers.

= Evaluation

After each training epoch we evaluate on the held-out test set with dropout disabled and no gradient computation. Test accuracy is $arg max_c p_c$ compared to the ground truth label, averaged over the full 10,000 test images. We track training loss (per batch, averaged per epoch), training accuracy, test accuracy, and test loss. The best checkpoint by test accuracy is saved and restored at the end of training.

The baseline configuration is Adam ($eta = 0.001$, default betas), no learning rate scheduler, standard width (32, 64, 128 filters), ReLU, data augmentation, and no regularization. This hits *78.15%* test accuracy after 15 epochs, clearing the 75% requirement.

= Ablation Study

We ran 8 ablation experiments. Each one varies a single factor while holding everything else at the baseline (Adam, $eta = 0.001$, $beta_1 = 0.9$, $beta_2 = 0.999$, no scheduler, standard width, ReLU, data augmentation, no regularization). All ablation runs use 15 epochs.

== Effect of Optimization Algorithm and Learning Rate

We compared Adam ($eta = 0.001$) against SGD with Momentum ($beta = 0.9$) at three learning rates: 0.1, 0.05, and 0.01.

#figure(
  image("images/exp4/exp4_optimizer_loss.png", width: 85%),
  caption: [Training loss over 15 epochs for Adam and SGD+Momentum at different learning rates.]
)

#figure(
  image("images/exp4/exp4_optimizer_test_acc.png", width: 85%),
  caption: [Test accuracy over 15 epochs for Adam and SGD+Momentum at different learning rates.]
)

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt,
    [*Optimizer*], [*LR*], [*Final Test Acc*], [*Final Train Loss*],
    [Adam],            [0.001], [78.15%], [0.5969],
    [SGD + Momentum],  [0.1],   [76.99%], [0.6548],
    [SGD + Momentum],  [0.05],  [73.66%], [0.7752],
    [SGD + Momentum],  [0.01],  [62.81%], [1.0870],
  ),
  caption: [Final results after 15 epochs for each optimizer and learning rate.]
)

*Findings.* Adam wins by a clear margin. By epoch 6 it is already at 72.0% while Momentum at lr=0.1 is at 70.1%, and by epoch 15 Adam leads by 1.2 points. The gap comes from Adam's per-parameter adaptive learning rates: a single $eta = 0.001$ just works, while Momentum is highly sensitive to the global learning rate. At lr=0.1 Momentum is competitive; at lr=0.05 it falls 4.5 points behind; at lr=0.01 it barely converges.

*Takeaway.* Adam is the better choice for this task and epoch budget. We use Adam at $eta = 0.001$ for all remaining experiments.

== Effect of Adam Beta Parameters

We swept $beta_1 in {0.8, 0.9, 0.95}$ with $beta_2 = 0.999$ fixed, and $beta_2 in {0.9, 0.99, 0.999}$ with $beta_1 = 0.9$ fixed.

#figure(
  image("images/exp5/exp5_adam_betas_loss.png", width: 85%),
  caption: [Training loss over 15 epochs for each Adam beta configuration.]
)

#figure(
  image("images/exp5/exp5_adam_betas_test_acc.png", width: 85%),
  caption: [Test accuracy over 15 epochs for each Adam beta configuration.]
)

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt,
    [*Config*], [*$beta_1$*], [*$beta_2$*], [*Final Test Acc*],
    [Default],           [0.9],  [0.999], [76.61%],
    [Higher $beta_1$],   [0.95], [0.999], [75.70%],
    [Lower $beta_1$],    [0.8],  [0.999], [77.65%],
    [Lower $beta_2$],    [0.9],  [0.99],  [77.48%],
    [Much lower $beta_2$], [0.9], [0.9],  [75.75%],
  ),
  caption: [Final test accuracy after 15 epochs for each beta configuration.]
)

*Findings.* Lowering $beta_1$ from 0.9 to 0.8 gains about 1 point. A smaller $beta_1$ gives less weight to past gradients in the first moment, so the optimizer responds faster to the current gradient. That matters early in training when the loss landscape is changing quickly. Pushing $beta_1$ up to 0.95 does the opposite: the optimizer over-smooths and misses sharp gradient signals.

For $beta_2$, reducing from 0.999 to 0.99 also helps slightly (77.48% vs. 76.61%). Faster adaptation of the second moment lets the per-parameter learning rates update more responsively. Going to 0.9 is too aggressive and introduces noise.

*Takeaway.* Both $beta_1 = 0.8$ and $beta_2 = 0.99$ individually beat the defaults. We use both in the final configuration.

== Effect of Learning Rate Decay

We compared three schedules on top of Adam: cosine decay to zero over 15 epochs, step decay ($gamma = 0.5$ every 5 epochs), and no decay.

#figure(
  image("images/exp1/exp1_lr_decay_loss.png", width: 85%),
  caption: [Training loss over 15 epochs for each learning rate schedule.]
)

#figure(
  image("images/exp1/exp1_lr_decay_test_acc.png", width: 85%),
  caption: [Test accuracy over 15 epochs for each learning rate schedule.]
)

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt,
    [*Schedule*], [*Final Test Acc*], [*Peak Test Acc*], [*Final Train Loss*],
    [Cosine Decay], [75.46%], [75.46%], [0.6466],
    [Step Decay],   [77.10%], [77.65%], [0.6206],
    [No Decay],     [77.53%], [77.53%], [0.6306],
  ),
  caption: [Results after 15 epochs for each learning rate schedule.]
)

*Findings.* Cosine decay falls about 2 points behind the other two. The problem is that cosine decay starts reducing the learning rate immediately, right when the model is still far from any reasonable optimum. The loss curves show cosine lagging for the first 8 epochs, and by the time a small learning rate would actually be useful for fine-tuning, it has already burned through its budget.

Step decay and no decay perform comparably here, which makes sense at only 15 epochs. The model has not started oscillating around the minimum, so a constant learning rate is fine in the short term. The brief accuracy plateaus at epochs 5 and 10 visible in the step decay curves are the scheduler kicks landing.

*Takeaway.* Keep the learning rate high early in training. Step decay is the better long-term choice since it has a mechanism for late-stage refinement, but both beat cosine at 15 epochs.

== Effect of Data Augmentation

We trained with and without augmentation. The augmented pipeline adds random horizontal flip and random crop with padding 4. The test pipeline is the same in both cases.

#figure(
  image("images/exp6/exp6_augmentation_loss.png", width: 85%),
  caption: [Training loss over 15 epochs with and without data augmentation.]
)

#figure(
  image("images/exp6/exp6_augmentation_test_acc.png", width: 85%),
  caption: [Test accuracy over 15 epochs with and without data augmentation.]
)

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt,
    [*Augmentation*], [*Final Test Acc*], [*Final Train Acc*], [*Final Train Loss*],
    [Enabled],  [78.06%], [78.21%], [0.6269],
    [Disabled], [64.86%], [98.62%], [0.0436],
  ),
  caption: [Results after 15 epochs with and without data augmentation.]
)

*Findings.* Without augmentation, training accuracy hits 98.62% by epoch 5 while test accuracy is stuck at 65% and going nowhere. Training loss collapses to near zero while test loss climbs past 2.0. Classic overfit: the model is memorizing specific training images rather than learning anything generalizable. With augmentation, training and test accuracy stay close to each other throughout all 15 epochs.

This is the biggest single factor in the ablation study, more than 12 percentage points difference.

*Takeaway.* Augmentation is not optional here. Without it the model memorizes the training set. We use it in all experiments.

== Effect of Network Width

We compared three filter configurations: slim (16, 32, 64), standard (32, 64, 128), and wide (64, 128, 256).

#figure(
  image("images/exp7/exp7_width_loss.png", width: 85%),
  caption: [Training loss over 15 epochs for standard and wide network configurations.]
)

#figure(
  image("images/exp7/exp7_width_test_acc.png", width: 85%),
  caption: [Test accuracy over 15 epochs for standard and wide network configurations.]
)

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt,
    [*Width*], [*Filters*], [*Final Test Acc*], [*Final Train Loss*],
    [Slim],     [(16, 32, 64)],   [75.44%], [0.7128],
    [Standard], [(32, 64, 128)],  [77.18%], [0.6338],
    [Wide],     [(64, 128, 256)], [78.17%], [0.6305],
  ),
  caption: [Results after 15 epochs for each width configuration.]
)

*Findings.* There is a clear trend: more filters, better accuracy. Slim hits 75.44%, standard 77.18%, wide 78.17%. Each doubling of width adds roughly 1.5 to 1.7 percentage points. More filters means more feature maps at each stage, so the network can represent a richer set of patterns. The gap is modest at 15 epochs because none of the models have fully saturated their capacity yet. Over longer training the wider model's extra capacity would matter more.

*Takeaway.* Wider filters are a consistent improvement. We use (64, 128, 256) for the final model.

== Effect of Activation Function

We swapped ReLU for Sigmoid in all conv and fully connected layers, keeping Softmax for the output.

#figure(
  image("images/exp8/exp8_activation_loss.png", width: 85%),
  caption: [Training loss over 15 epochs for ReLU and Sigmoid activations.]
)

#figure(
  image("images/exp8/exp8_activation_test_acc.png", width: 85%),
  caption: [Test accuracy over 15 epochs for ReLU and Sigmoid activations.]
)

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt,
    [*Activation*], [*Final Test Acc*], [*Final Train Loss*], [*Notes*],
    [ReLU],    [77.78%], [0.6260], [Converges cleanly],
    [Sigmoid], [67.23%], [0.9501], [Slow, gradient-starved],
  ),
  caption: [Results after 15 epochs for each activation function.]
)

*Findings.* Sigmoid loses by over 10 points. The problem is vanishing gradients: Sigmoid saturates at both extremes of its range, where the derivative goes to zero. With multiple saturated layers, the gradient product shrinks exponentially by the time it reaches the early layers, which get almost no training signal. ReLU does not have this problem since its gradient is exactly 1 for all positive inputs, so gradients flow through unchanged.

He initialization (which we use) is also designed for ReLU specifically. It accounts for the fact that ReLU kills half the neurons on average and scales the initial weights to compensate. That assumption does not hold for Sigmoid, so the initialization is a bad fit and makes convergence worse.

*Takeaway.* ReLU is the right choice. Sigmoid is not suitable for networks of this depth with He initialization.

== Effect of Regularization Method

We compared four configurations: no regularization, L2 only ($lambda = 0.001$), Dropout only ($p = 0.3$), and both combined.

#figure(
  image("images/exp2/exp2_regularization_loss.png", width: 85%),
  caption: [Training loss over 15 epochs for each regularization configuration.]
)

#figure(
  image("images/exp2/exp2_regularization_test_acc.png", width: 85%),
  caption: [Test accuracy over 15 epochs for each regularization configuration.]
)

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt,
    [*Regularization*], [*Final Test Acc*], [*Final Train Acc*], [*Final Train Loss*],
    [None],          [77.78%], [77.88%], [0.6311],
    [L2 only],       [73.01%], [70.79%], [0.8334],
    [Dropout only],  [74.23%], [70.70%], [0.8334],
    [L2 + Dropout],  [71.65%], [66.36%], [0.9474],
  ),
  caption: [Results after 15 epochs for each regularization strategy.]
)

*Findings.* No regularization wins at 15 epochs, which is expected. Regularization trades training performance for generalization, but that trade only pays off once the model is actually overfitting. At this epoch count and model size, the unregularized model has not overfit yet, so the regularization just slows it down.

All three regularized variants show higher training loss and lower training accuracy, which is the regularization working as intended. But it does not translate to better test accuracy here. Dropout outperforms L2 by about 1.2 points, and combining both is worse than either alone since the compounding constraints hit convergence speed harder.

*Takeaway.* Regularization is most valuable in longer training runs where overfitting becomes real. For the final model we skip it, since 50 epochs is still a short run for this architecture.

== Effect of L2 Regularization Strength

We swept the L2 penalty $lambda$ over four orders of magnitude: $10^(-4)$, $10^(-3)$, $10^(-2)$, $10^(-1)$.

#figure(
  image("images/exp3/exp3_l2_lambda_loss.png", width: 85%),
  caption: [Training loss over 15 epochs for each $lambda$ value.]
)

#figure(
  image("images/exp3/exp3_l2_lambda_test_acc.png", width: 85%),
  caption: [Test accuracy over 15 epochs for each $lambda$ value.]
)

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt,
    [*$lambda$*], [*Final Test Acc*], [*Final Train Loss*], [*Observation*],
    [$10^(-4)$], [77.35%], [0.6435], [Near-baseline],
    [$10^(-3)$], [73.80%], [0.8140], [Mild constraint],
    [$10^(-2)$], [51.49%], [1.4095], [Over-regularized],
    [$10^(-1)$], [23.74%], [2.0706], [Training failed],
  ),
  caption: [Results after 15 epochs across L2 lambda values.]
)

*Findings.* The effect is highly nonlinear. At $lambda = 10^(-4)$ the model behaves almost identically to no regularization: 77.35% test accuracy, training loss basically the same. The penalty gradient $2 lambda w$ is too small to compete with the data gradient.

At $lambda = 10^(-3)$ we lose about 4 points. The penalty is now large enough to meaningfully shrink weights and slow learning, and since the model is not actually overfitting, this is purely a downside.

At $lambda = 10^(-2)$ training breaks. Accuracy plateaus at 51% and barely moves. The penalty gradient has taken over, so weight updates are dominated by weight shrinkage rather than the actual loss signal.

At $lambda = 10^(-1)$ the model learns nothing. Test accuracy stays near random (24%) and loss never drops meaningfully.

*Takeaway.* The usable range is narrow. Below $10^(-4)$ does nothing; at or above $10^(-2)$ training fails. If L2 were needed, $lambda = 10^(-4)$ is the sweet spot for this architecture and learning rate.

= Final Training with Optimal Configuration

Based on the ablation results, we assembled the best combination of settings:

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    stroke: 0.5pt,
    [*Component*], [*Choice*],
    [Activation],     [ReLU throughout, Softmax on output],
    [Optimizer],      [Adam, $eta = 0.001$, $beta_1 = 0.8$, $beta_2 = 0.99$, $epsilon = 10^(-8)$],
    [LR Schedule],    [Step decay: $gamma = 0.5$ every 5 epochs],
    [Regularization], [None],
    [Augmentation],   [Random horizontal flip + random crop (padding 4)],
    [Epochs],         [50],
  ),
  caption: [Final training configuration.]
)

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: 0.5pt,
    [*Layer*], [*Input Shape*], [*Output Shape*],
    [Conv(3 #sym.arrow.r 64, k=3, s=1, p=1) + ReLU],    [(N, 3, 32, 32)],   [(N, 64, 32, 32)],
    [Conv(64 #sym.arrow.r 128, k=3, s=2, p=1) + ReLU],  [(N, 64, 32, 32)],  [(N, 128, 16, 16)],
    [Conv(128 #sym.arrow.r 256, k=3, s=2, p=1) + ReLU], [(N, 128, 16, 16)], [(N, 256, 8, 8)],
    [Flatten], [(N, 256, 8, 8)], [(16384, N)],
    [FC(16384 #sym.arrow.r 512) + ReLU], [(16384, N)], [(512, N)],
    [FC(512 #sym.arrow.r 10) + Softmax], [(512, N)],   [(10, N)],
  ),
  caption: [Final model architecture.]
)

#figure(
  image("images/final/final_loss.png", width: 85%),
  caption: [Training loss over 50 epochs with the optimal configuration.]
)

#figure(
  image("images/final/final_train_acc.png", width: 85%),
  caption: [Training accuracy over 50 epochs with the optimal configuration.]
)

#figure(
  image("images/final/final_test_acc.png", width: 85%),
  caption: [Test accuracy over 50 epochs with the optimal configuration.]
)

We trained for 50 epochs and restored the best checkpoint at the end. The model crosses 80% test accuracy at epoch 13 and peaks at *82.51%* at epoch 48. After epoch 30 the learning rate has decayed enough that improvements become incremental, and test accuracy settles in a narrow band above 82%.

The final 82.51% clears both the 75% baseline requirement and the ablation baseline of 78.15%, a gain of 4.36 points from wider filters, better Adam betas, and longer training.

= Individual Contributions

#table(
  columns: (auto, 1fr),
  align: (left, left),
  stroke: 0.5pt,
  [*Member*], [*Contributions*],
  [Anton Sakhanovych], [
    Implemented all layer forward and backward passes from scratch: ReLU, Softmax, Convolutional (He init, im2col via F.unfold, matmul-based forward/backward), Flatten, Dropout (inverted, training/eval mode), and Fully Connected. Implemented SGD, SGD with Momentum, and Adam optimizers; L2 regularizer; Cosine and Step learning rate schedulers. Wrote unit tests and numerical gradient checks for all layers. Set up GitHub Actions CI with ruff formatting checks and pytest runs on push and pull request.
  ],
  [Gavin D'Hondt], [
    Set up the project repository with uv and pyproject.toml. Restructured the activation abstraction to compose on the Layer protocol. Implemented the CIFAR-10 data loading pipeline with per-channel normalization, training augmentation, and separate train/test loaders. Built the training and evaluation loop in train.py with per-epoch history tracking and best-checkpoint saving. Designed and implemented the CNN model. Rewrote the convolutional layer with F.unfold and in-place operations to resolve OOM issues during Colab training. Integrated all components into a working end-to-end pipeline. Built the ablation experiment runner (run_ablation.py) with argument parsing, result serialization, and plot generation. Identified and applied the optimal hyperparameter configuration for the final run.
  ],
)
