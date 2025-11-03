# KulingNet
Ground-Based Cloud Classiﬁcation With Deep CNN, based on CCSN dataset

## KulingNet1.0:
- Basic CNN, 5x5 kernel
- 3 Conv-layers, 3 fc-layers
- 

## KulingNet2.0:
- 4 Conv-layers, 2 fc-layers
- Adaptive pooling, dropout, batchNorm2d
- Track with Confusion Matrix, classification report
-  

**Next steps**: 
- Modify optimizer metrics: learning rate + momentum?
- ~~Plot validation/training loss~~
- ~~Refine CNN through confusion matrix~~
- More advanced CNN? (more layers/parameters)

## KulingNet3.0:
- 

**Next steps**: 
- Transfer learning (Utilizing pre-trained CNN, ex. ResNet-18 or EfficientNet)
- Follow structure of CloudNet (from article)
    Article it is based on:
    https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2018GL077787


------------------------------------------------------------------------

Notes on CCSN:
The CCSN dataset contains 2543 cloud images. 
According to the World Meterological Organization’s genera-based 
classification recommendation, we divide into 11 different categories：

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

Link to examples:
- https://www.kaggle.com/code/tamoshreedey/coudcustomcnn
- 


ReadMe formatting: https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax 


![alt text](https://github.com/ThomasM31/KulingNet/blob/main/CCSN_Image_set/Cu/Cu-N009.jpg?raw=true)
