# VGG16-with-AveragePooling
Here, we custom train VGG16 network with AveragePooling layer instead of MaxPooling layer
The default architecture of VGG16 uses MaxPooling layers in each blocks. However, it has been found that MaxPooling 
causes overfitting problems and also results in poorer accuracy for image sizes greater than 224x224 as input. 
Here we customize the architecure by making use of an addition layer before 'flatten' layer. We use 
AveragePooling2D layer just bore the flatten layer of size 7x7 and then pass it to the flatten layer. In our tests,
we have found it to prform better than the default architecture of VGG16.
