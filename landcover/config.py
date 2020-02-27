# Number of target classes
HR_NCLASSES = 5
LR_NCLASSES = 22

# Positional index of labels in patches
HR_LABEL_INDEX = 8
LR_LABEL_INDEX = 9

# Keys for transformation of labels
HR_LABEL_KEY = "data/cheaseapeake_to_hr_labels.txt"
LR_LABEL_KEY = "data/nlcd_to_lr_labels.txt"

# LR files used for superres loss
LR_STATS_MU = "data/nlcd_mu.txt"
LR_STATS_SIGMA = "data/nlcd_sigma.txt"
LR_CLASS_WEIGHTS = "data/nlcd_class_weights.txt"
