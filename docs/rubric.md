CSI 4140/5140 Deep learning and applications

*Due by 03/22/2026 11:59:59pm*

### Summary

Develop and train a deep neural network (DNN) using PyTorch to classify
images from the CIFAR-10 dataset. Perform a comprehensive ablation study
to assess the impact of various methods and hyperparameters on model
performance. To deepen your understanding of neural networks, you'll
have to manually implement it without using PyTorch's built-in
functions.

This project should be completed in groups of 1-2 people. I will not
assign group members and you need to find your group member by yourself.

Sign up your group in the Google sheet: [[CSI 4140/5140 - Winter 2026 -
Project 1 -
Group]{.underline}](https://docs.google.com/spreadsheets/d/1ep7dTJC0SAbBphuNdnwH2fGF0gXf4BCcrMl_Bjcmcdc/edit?usp=sharing)You
still need to sign up even if you do it individually.

You can use Google Colab to train the model. Google Colab supports GPU.
To use GPU to train the model, refer to the tutorials in the Pytorch
official website, like:
[[https://pytorch.org/tutorials/beginner/pytorch_with_examples.html]{.underline}](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)

Grading will be based on the code, report, and presentation.

### Description

**1. Model Architecture**: Your model must include the following layers,
which you'll have to manually implement without using PyTorch's built-in
functions.

1)  At least one fully-connected layer

2)  At least one convolutional layer

3)  At least one ReLU activation layer

4)  Softmax layer (for output classification)

2\. **Performance requirement**: The final test accuracy of your model
should exceed 75% on the CIFAR-10 dataset. The higher, the better. You
will get extra credit if the accuracy exceeds 80%.

3\. **Regularization Techniques**: The following regularization methods
should be implemented without using PyTorch's built-in functions.

1)  L2 regularization

2)  Dropout

The above techniques should be implemented and evaluated in the ablation
study, but not necessarily used in your model training for getting a
good test accuracy.

Optional: You can use some data augmentation methods (e.g., random
cropping, flipping, etc.).

4\. **Optimization Techniques:** The following optimization methods
should be implemented without using PyTorch's built-in functions.

1)  Gradient descent with momentum

2)  Adam

3)  Cosine learning rate decay

4)  One additional learning rate decay algorithm

The above techniques should be implemented and evaluated in the ablation
study, but not necessarily used in your model training for getting a
good test accuracy.

5\. **Evaluation Metrics & Visualizations**: Plot the figures about the
following metrics of the model achieving over 75% accuracy.

1)  Training accuracy over each epoch

2)  Test accuracy over each epoch

3)  Cost over each iteration

6\. **Ablation Study**: Conduct a thorough ablation study to analyze the
effects of different methods and hyperparameters on model performance.
You are required to study and report the effects on the test accuracy
(required), and the following metrics (optional), including convergence
speed, training accuracy, cost values, etc.

Findings or insights should be discussed. Virtualization is preferred,
like figures of accuracy over epoch. Convergence speed refers to the
rate at which a model's training process reduces the loss function and
approaches an optimal or acceptable performance on the training data.

Your ablation study should include, at a minimum, the following aspects
(the more aspects, the better). To save training time, for the ablation
study, you do not need to train the model for the entire epochs; you can
only train the model for a few epochs---just enough to observe
hyperparameter differences.

1)  Effect of at least two different learning rate decay algorithms

2)  Effect of different regularization methods, at least including L2
    regularization and Dropout

3)  Effect of lambda of L2 regularization

4)  Effect of different optimization algorithms, at least including
    gradient descent with momentum and Adam

5)  Effect of beta1 and beta2 of Adam

Ablation study is a crucial element in research papers, providing
valuable insights into the contribution of each component of a model or
system. About how to conduct ablation study, refer to the Section of
Ablation Study in the following papers. You can refer to more papers
published in the top AI conferences, e.g., NeurIPS, ICML, ICLR, CVPR,
ICCV, ECCV, AISTATS.

- Hrank: Filter pruning using high-rank feature map:
  [[https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_HRank_Filter_Pruning_Using_High-Rank_Feature_Map_CVPR_2020_paper.pdf]{.underline}](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_HRank_Filter_Pruning_Using_High-Rank_Feature_Map_CVPR_2020_paper.pdf)

- Contrastive representation distillation:
  [[https://arxiv.org/pdf/1910.10699]{.underline}](https://arxiv.org/pdf/1910.10699)

**Important Notes:**

You are required to manually implement all the required layers and
algorithms without using PyTorch\'s built-in functions. I will review
your code, and may use a script to verify that no built-in functions are
used. If any prohibited built-in functions are found, you will lose all
points associated with those implementations.

For more information about the built-in functions, refer to: [[Pytorch
built-in
functions]{.underline}](https://docs.google.com/document/d/11gnG5m6NwZ06NWWopohfBv-0-e7DisUozxRFAtxjd28/edit?usp=sharing)

The following built-in data loader function, data augmentation
functions, pooling layers, and normalization layers are allowed to use:

- torch.utils.data.DataLoader

- torchvision.transforms.RandomCrop

- torchvision.transforms.RandomHorizontalFlip

- torchvision.transforms.RandomVerticalFlip

- torchvision.transforms.ColorJitter

- torchvision.transforms.RandomResizedCrop

- Pooling Layers

  - torch.nn.functional.max_pool1d

  - torch.nn.functional.max_pool2d

  - torch.nn.functional.max_pool3d

  - torch.nn.functional.avg_pool1d

  - torch.nn.functional.avg_pool2d

  - torch.nn.functional.avg_pool3d

  - torch.nn.functional.adaptive_max_pool2d

  - torch.nn.functional.adaptive_avg_pool2d

- Normalization Layers

  - torch.nn.functional.batch_norm

  - torch.nn.functional.instance_norm

  - torch.nn.functional.layer_norm

  - torch.nn.functional.group_norm

### Suggested Steps to Finish the Project

Start Early. Do not expect that you can finish everything in the last
week.

Step 1 (Week 1): Implement a FC-NN with fully-connected layers, ReLU
activation layers, and softmax layer, as well as observe and plot its
training accuracy, test accuracy, and cost value.

Step 2 (Week 2): Implement and apply different regularization and
optimization techniques to your model.

Step 3 (Week 3): Implement convolutional layers to your model, and
adjust the model architectures to achieve a 75% above accuracy.

Step 4 (Week 4): Conduct ablation study, summarize your results, and
write the report.

### Submission

Project submission includes two steps.

**Step 1:** You need to submit your implementation on Moodle by
03/22/2026 11:59:59pm. The submission link will be closed immediately
after the deadline. Your submission should be a single zip file that is
named by the last names of your team members. You may use this naming
structure: \<groupXX_lastname1_lastname2\>.zip.

The zip file should contain the following:

- A folder containing all your code.

- A simple README file that describes the structure of your code and how
  to run it.

- A PDF report. Use one of the provided templates to prepare your
  report.

  - Google Doc template: [[CSI 4140/5140 --- Project1
    Report]{.underline}](https://docs.google.com/document/d/1rybKpNbSs5PrII63gGqffBeGNKyVqvmLEZ7Kb8SYLUM/edit?usp=sharing)

  - Overleaf template:
    [[https://www.overleaf.com/read/vxczqfmvpqhp#254df5]{.underline}](https://www.overleaf.com/read/vxczqfmvpqhp#254df5)

Do not include any binary file in your zip file. Only one submission is
needed per group.

Failure to follow the above instructions to prepare your submission will
cause a penalty to your grade.

**Step 2**: You need to present your project in class, 03/24/2026
7:30pm. You have about 5 minutes for the presentation. The presentation
duration may be adjusted before one week of the deadline according to
the number of groups.

### Policies

1)  Late submissions: refer to the late policy in the syllabus.

2)  Each group needs to work independently on this exercise. We
    encourage high-level discussions among groups to help each other
    understand the concepts and principles. However, code-level
    discussion is prohibited and plagiarism will directly lead to
    failure of this course.
