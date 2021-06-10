# CheXNet-for-Augmented-Pneumonia

A version of CheXNet was implemented to diagnose Pneumonia based on frontal X-ray images.

The two Generative Adversarial Networks (GANs), namely CGAN and DCGAN, are used to generate more data for the imbalanced class (Negative). The dataset folder should be organized as the followings:
---datasets |
                ---train |
                             --- NORMAL
                             --- NEUMONIA
                ---test  |
                             --- NORMAL 
                             --- NEUMONIA
                ---augmented datasets 1 |
                             --- NORMAL
                             --- NEUMONIA
                ---augmented datasets 2 |
                             --- NORMAL
                             --- NEUMONIA
In CheXNet, there are options to choose the portions you want from the original and each generated dataset. They are set by flags:
--real_train_portion=0.3
--aug_train_portion=0.6
In this case, 30% and 60% of samples will be randomly picked from the original and generated datasets.
