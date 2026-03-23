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

We are classifying 32 #sym.times 32 RGB images from the CIFAR-10 dataset into 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. There are 50,000 training images and 10,000 test images.

The constraint is that all layers, optimizers, and schedulers must be implemented from scratch in PyTorch, without using any built-in `nn.Module` layers or optimizer classes. This means we had to work through the math directly and write the forward and backward passes ourselves rather than relying on pre-built abstractions.

= Design and Implementation

== Model Architecture

Our model is a three-block CNN with two fully connected layers on top. Rather than using pooling layers for spatial downsampling, we use strided convolutions, which let the network learn how to downsample rather than just taking the max or average of a patch.

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

Below is a breakdown of how we implemented each component:

- *Convolutional layer:* At its core, a convolution is a dot product between the kernel and each local patch of the input. The naive way to do this is to loop over every spatial position, but that is very slow. Instead, we use `F.unfold` to extract all the patches and lay them into a single matrix. For each of the $H_"out" dot W_"out"$ positions, it grabs the $C_"in" dot K^2$ values the kernel would see and lines them up as a column, giving shape $(N, C_"in" dot K^2, H_"out" dot W_"out")$. From there, one matrix multiply with the weight matrix (reshaped to $(C_"out", C_"in" dot K^2)$) computes every output position at once. For the backward pass, the weight gradient is $nabla_"mat" dot X_"unfolded"^top$ summed over the batch, and the input gradient is $W^top dot nabla_"mat"$, which gives per-patch error signals in column form. Then `F.fold` adds up all the overlapping contributions and reconstructs the original spatial layout.

- *Activation layers:* ReLU is $(x + |x|) / 2$; the backward pass just multiplies the incoming gradient by $(x > 0)$. Softmax subtracts the per-sample max before exponentiating (for numerical stability) and then normalizes column-wise.

- *Flatten layer:* Reshapes $(N, C, H, W)$ to $(C dot H dot W, N)$ so it fits the column-major convention our fully connected layers expect. We cache the input shape so the backward pass can undo the reshape.

- *Fully connected layer:* Forward is $Z = W X + b$ followed by the activation. Backward: $nabla W = nabla Z dot X^top$, $nabla b = sum_"batch" nabla Z$, $nabla X = W^top dot nabla Z$.

- *Dropout layer:* We use inverted dropout. During training, a binary mask is sampled with keep probability $(1 - p)$ and immediately scaled by $1 / (1 - p)$ so that the expected output value stays the same as the input. The same mask is reused in the backward pass, and during evaluation dropout is skipped.

== Methods

*Loss function.* We use cross-entropy over the softmax probabilities:
$cal(L) = -1/N sum_(i=1)^N log p_(y_i)$,
where $p_(y_i)$ is the predicted probability for the correct class. The gradient works out to $-1/(N p_(y_i))$ at the true class index and zero everywhere else.

*Optimizers.* We implemented three optimizers, all sharing a `BaseOptimizer` base class that handles parameter iteration and optional regularization:

- *SGD:* $w arrow.l w - eta nabla_w cal(L)$

- *SGD + Momentum (EMA style):* Maintains a velocity per parameter: $v arrow.l beta v + (1 - beta) nabla_w cal(L)$, then $w arrow.l w - eta v$. We used the EMA formulation because it keeps $v$ in the same magnitude range as the gradient, unlike the standard momentum formulation which accumulates unboundedly.

- *Adam:* $m arrow.l beta_1 m + (1 - beta_1) g$, $v arrow.l beta_2 v + (1 - beta_2) g^2$, bias-corrected as $hat(m) = m / (1 - beta_1^t)$, $hat(v) = v / (1 - beta_2^t)$, then $w arrow.l w - eta hat(m) / (sqrt(hat(v)) + epsilon)$. We used the standard defaults: $beta_1 = 0.9$, $beta_2 = 0.999$, $epsilon = 10^(-8)$.

*Learning rate schedules.* We implemented two. Cosine decay follows:
$eta_t = eta_"min" + 1/2 (eta_0 - eta_"min")(1 + cos(pi t / T))$.
Step decay multiplies the current learning rate by $gamma$ every $k$ epochs, with a floor at $eta_"min"$.

*Regularization.* Our L2 regularizer adds $2 lambda w$ to the weight gradient during the optimizer step, and it only touches weight parameters, not biases. We made the regularizer its own protocol class, decoupled from the optimizer, so we could swap regularization strategies without having to dig into the optimizer code. Dropout is applied after each convolutional activation during training.

*Data pipeline.* All images get normalized per channel using the standard CIFAR-10 statistics: mean $(0.4914, 0.4822, 0.4465)$, std $(0.2470, 0.2435, 0.2616)$. For training, we apply random horizontal flips and random crops with padding 4. The test loader has no augmentation and is fully deterministic.

*Weight initialization.* We use He initialization for both convolutional and fully connected weights: $w tilde cal(N)(0, 2/n_"in")$, where $n_"in"$ accounts for kernel area in the conv layers. This is specifically designed for ReLU, which matters later when we try Sigmoid.

= Evaluation

After every training epoch, we run the model on the held-out test set with dropout turned off and no gradient computation. We measure test accuracy by comparing $arg max_c p_c$ against the ground truth label, averaged across all 10,000 test images. Throughout training we track training loss (per batch, averaged per epoch), training accuracy, test accuracy, and test loss. And at the end, we restore whichever checkpoint had the best test accuracy.

Our baseline configuration (Adam at $eta = 0.001$ with default betas, no learning rate scheduler, standard width of 32, 64, 128 filters, ReLU activations, data augmentation on, and no regularization) lands at *78.15%* test accuracy after 15 epochs, clearing the 75% requirement.

= Ablation Study

For the ablation, we ran 8 experiments. Each experiment changes exactly one thing while holding everything else at the baseline described above (Adam, $eta = 0.001$, $beta_1 = 0.9$, $beta_2 = 0.999$, no scheduler, standard width, ReLU, augmentation on, no regularization). Every ablation run goes for 15 epochs.

== Effect of Optimization Algorithm and Learning Rate

The first thing we wanted to understand was how much the optimizer matters. We compared Adam ($eta = 0.001$) against SGD with Momentum ($beta = 0.9$) at three different learning rates: 0.1, 0.05, and 0.01.

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

Adam wins by a clear margin. By epoch 6, Adam is at 72.0% while the best Momentum configuration (lr=0.1) is trailing at 70.1%. By epoch 15, Adam leads by 1.2 points. The gap comes from Adam's per-parameter adaptive learning rates: a single $eta = 0.001$ works well, whereas Momentum is very sensitive to the choice of learning rate. At lr=0.1 Momentum is competitive; at lr=0.05 it falls 4.5 points behind; and at lr=0.01 it barely converges, only reaching 62.81%.

Given these results, we stuck with Adam at $eta = 0.001$ for every remaining experiment.

== Effect of Adam Beta Parameters

We swept $beta_1 in {0.8, 0.9, 0.95}$ with $beta_2 = 0.999$ held constant, and separately swept $beta_2 in {0.9, 0.99, 0.999}$ with $beta_1 = 0.9$ held constant.

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

Dropping $beta_1$ from 0.9 to 0.8 picked up about a full percentage point. A smaller $beta_1$ puts less weight on past gradients when computing the first moment, so the optimizer reacts faster to what the gradient is doing at any given step. That matters early on when the loss landscape is shifting quickly. Going the other direction and pushing $beta_1$ up to 0.95 hurt performance because the optimizer was over-smoothing and missing sharper gradient signals.

On the $beta_2$ side, dropping from 0.999 to 0.99 also helped (77.48% vs. 76.61%). The second moment adapts faster, which means the per-parameter learning rates adjust more responsively. But going all the way down to 0.9 was too aggressive and introduced noise.

Since both $beta_1 = 0.8$ and $beta_2 = 0.99$ individually beat the defaults, we decided to use both together in the final configuration.

== Effect of Learning Rate Decay

We tested three learning rate schedules on top of Adam: cosine decay (annealing to zero over 15 epochs), step decay ($gamma = 0.5$ every 5 epochs), and just keeping the learning rate constant.

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

Cosine decay underperformed by about 2 points, and looking at the loss curves it is pretty clear why. Cosine starts pulling the learning rate down immediately from epoch 1, right when the model is still nowhere near a good optimum. It lags behind the other two for the first 8 epochs or so, and by the time a small learning rate would actually be helpful for fine-tuning, it has already used up most of its budget.

Step decay and no decay performed about the same here, which makes sense given that we only ran 15 epochs. The model has not yet started oscillating around a minimum, so there is no real need to reduce the learning rate in the short term. The step decay scheduler kicking in is visible at epochs 5 and 10, where there are brief plateaus in accuracy.

For the final model, we went with step decay. At 15 epochs it does not make a difference, but for the longer 50-epoch run we planned, having a mechanism to ramp down the learning rate later on is important.

== Effect of Data Augmentation

We trained the model with and without augmentation. The augmented pipeline adds random horizontal flips and random crops with padding 4; the test pipeline is identical in both cases.

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

Without augmentation, the model memorizes the training set. Training accuracy hits 98.62% while test accuracy stalls around 65%, and training loss collapses to near zero while test loss climbs past 2.0. The network is just memorizing specific pixel patterns in the training images rather than learning features that generalize. With augmentation on, training and test accuracy stay close to each other throughout all 15 epochs.

This turned out to be the single biggest factor in the entire ablation study, accounting for more than 12 percentage points of difference. Augmentation is not optional for this model. We keep it on for everything.

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

The trend is consistent: more filters means better accuracy. Slim gets 75.44%, standard gets 77.18%, wide gets 78.17%. Each doubling of the filter count picks up roughly 1.5 to 1.7 percentage points. More feature maps at each stage means the network can represent a wider variety of patterns, and CIFAR-10 has enough class diversity that the extra capacity helps. The gap is still modest at 15 epochs because none of the models have fully saturated their capacity yet, but over a longer training run, the wider model's advantage would likely grow.

We used the wide configuration (64, 128, 256) for the final model.

== Effect of Activation Function

We swapped out ReLU for Sigmoid across all conv and fully connected layers, keeping Softmax on the output layer.

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

Sigmoid falls over 10 points behind ReLU due to vanishing gradients. Sigmoid saturates at both extremes of its output range, and when it saturates, the derivative goes to essentially zero. With multiple saturated layers, the gradient product shrinks exponentially by the time it reaches the early layers of the network, so those layers get almost no training signal. ReLU does not have this issue because its gradient is exactly 1 for all positive inputs, so gradients pass through without being squashed.

Additionally, He initialization (which is what we use) is designed specifically for ReLU. It accounts for the fact that ReLU zeroes out roughly half the neurons on average and scales the initial weights to compensate. That assumption does not hold for Sigmoid, so the initialization is a poor fit and makes the convergence problems worse.

ReLU is the clear choice here. We did not experiment with other activations like Leaky ReLU or GELU since the assignment only asked us to compare two.

== Effect of Regularization Method

We tested four setups: no regularization, L2 only ($lambda = 0.001$), Dropout only ($p = 0.3$), and both L2 and Dropout together.

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

No regularization wins at 15 epochs. Regularization trades training performance for generalization, but that trade only pays off once the model is actually overfitting. At this epoch count and model size, the unregularized model has not started overfitting yet; training and test accuracies are still within a point of each other. So the regularization is only slowing the model down without providing any benefit.

All three regularized variants have higher training loss and lower training accuracy, which means the regularization is constraining the weights as intended. But that constraint does not translate to better test accuracy because there is no overfitting to prevent. Dropout edged out L2 by about 1.2 points, and combining both was worse than either alone because the compounding constraints hit convergence speed too hard.

For the final model we skip regularization. Even at 50 epochs, this architecture with augmentation does not overfit badly enough to warrant it.

== Effect of L2 Regularization Strength

We also swept the L2 penalty $lambda$ across four orders of magnitude ($10^(-4)$, $10^(-3)$, $10^(-2)$, $10^(-1)$) to see how sensitive the model is to this hyperparameter.

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

The relationship is nonlinear. At $lambda = 10^(-4)$, the model behaves almost identically to having no regularization (77.35% test accuracy, training loss basically unchanged). The penalty gradient $2 lambda w$ is too small to compete with the data gradient.

At $lambda = 10^(-3)$ we lose about 4 points. The penalty is large enough to meaningfully shrink weights and slow down learning, but since the model is not overfitting, this is purely a downside.

At $lambda = 10^(-2)$, training breaks down. Accuracy plateaus around 51% and barely moves. The penalty gradient has taken over the update step, so weight updates are dominated by weight shrinkage rather than by the actual loss signal.

And at $lambda = 10^(-1)$, the model learns nothing. Test accuracy sits near random chance (24%) and the loss never meaningfully decreases.

The usable range for L2 is narrow. Below $10^(-4)$ has no effect; at or above $10^(-2)$ training fails. If L2 were needed for this architecture and learning rate, $10^(-4)$ would be the right value.

= Final Training with Optimal Configuration

Based on the ablation results, we assembled the best combination of settings from each experiment:

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

We trained for 50 epochs and restored the best checkpoint at the end. The model crosses 80% test accuracy around epoch 13 and peaks at *82.51%* at epoch 48. After about epoch 30 the learning rate has decayed enough that improvements become incremental, and test accuracy hovers in a narrow band above 82% for the last 20 epochs.

The final 82.51% clears both the 75% baseline requirement and our ablation baseline of 78.15%, a gain of 4.36 points from the combination of wider filters, tuned Adam betas, and the extra training time.

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
