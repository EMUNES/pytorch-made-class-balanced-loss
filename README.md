# pytorch-made-class-balanced-loss
A ready-to-use &amp; class-based-module for directly implementation of class balanced loss in pytorch

一个现成的，基于pytorch nn.Module类的class balanced loss实现

Thanks to the great work of vandit15 in [class-balanced-loss-pytorch](https://github.com/vandit15/Class-balanced-loss-pytorch). See more detailed info about class balanced loss in his git.

Also all credits to original authors and researchers working on the paper about class balanced loss. Check in the .py file to see their paper.

## reason for this repo
Class Balanced Loss worked pretty well for many class imbalanced datasets, especially when samples in the dataset are limited (below 10000). If the dataset is rather small, Class Balanced Loss can move focus to the small class in a very astonishing way (In one case the f1_score jumps from 0.0 to 0.3).

So I modified the codes from the original author and add some features for real-time usuage when building models using pytorch. Those codes help me through most situations. Hope this will help you too.
