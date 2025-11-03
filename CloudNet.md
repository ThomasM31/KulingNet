# CloudNet Information

CloudNet article: https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2018GL077787

- Optimized CNN based on Alexnet

![CloudNet Architecture](https://github.com/ThomasM31/KulingNet/blob/main/CloudNet_Architecture.png?raw=true)

- 5 conv-layers, 2 FC-layers
- Takes RGB-images, "subtract the mean red-green-blue value of each pixel over the training set to improvetraining speed and accuracy"
- 3rd & 4th layer connected directly without pooling (features are hard to extract)
- Dropout 
- SGD with defalut parameters (Krizhevsky et al., 2012)
- CNN is trained from scratch
- Images are augmented with random crop and flip
- "Trained with a stochastic gradient usingthe machine learning software package Caﬀe (Jia et al., 2014) and running on an NVIDIA GeForce GTX780Tiwith batch size 8."
- learning rate: 0.001, EPOCHS: 20,000, momentum: 0.9. Learning rate reduced by factor of 10 every 5,000 epochs
- 

![CloudNet Confusion Matrix](https://github.com/ThomasM31/KulingNet/blob/main/CloudNet_CM.png?raw=true)

