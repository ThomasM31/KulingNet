# KulingNet
Ground-Based Cloud Classiﬁcation With Deep CNN, based on CCSN dataset

![Example cloud image, Cu-N009.jpg](https://github.com/ThomasM31/KulingNet/blob/main/CCSN_Image_set/Cu/Cu-N009.jpg?raw=true)

Example image from CCSN dataset: (Cumulus, N009)

## KulingNet1.0:
- Basic CNN, 5x5 kernel
- 3 Conv-layers, 3 fc-layers

## KulingNet2.0:
- 4 Conv-layers, 2 fc-layers, 3x3 kernel
- Adaptive pooling, dropout, batchNorm2d
- Evaluate with Confusion Matrix, classification report

## KulingNet3.0:
(Follows structure of CloudNet (https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2018GL077787), see CloudNet.md for notes)
- 5 conv layers, 2 FC-layers
- Dropout to 5th & 6th layer after nonlinearity
- SGD with defalut parameters (Krizhevsky et al., 2012)
- CNN is trained from scratch
- learning rate: 0.001, momentum: 0.9. 
- Images are augmented with random crop and flip

------------------------------------------------------------------------

Notes on CCSN:
The CCSN dataset contains 2543 cloud images. According to the World Meterological Organization’s genera-based classification recommendation, we divide into 11 different categories：

Ci = cirrus; 
Cs = cirrostratus; 
Cc = cirrocumulus; 
Ac = altocumulus; 
As = altostratus; 
Cu = cumulus; 
Cb = cumulonimbus; 
Ns = nimbostratus; 
Sc = stratocumulus; 
St = stratus; 
Ct = contrail.


All images are fixed resolution 256×256 pixels with the JPEG format.

-----------------------------------------------------------------------

## Next steps: 
- Transfer learning (Utilizing pre-trained CNN, ex. ResNet-18 or EfficientNet)
- Check different articles for potential other structures??
- Scale up EPOCHS: 20,000 (currently EPOCHS=100) + reduce learning rate by factor of 10 every 5,000 epochs

ReadMe formatting: https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax 

