# V1-Nonlinear

Paper has been published: https://jov.arvojournals.org/article.aspx?articleid=2793712

Citation: Miao, H.-Y., & Tong, F. (2024). Convolutional neural network models applied to neuronal responses in macaque V1 reveal limited nonlinear processing. Journal of Vision, 24(6), 1-19. https://doi.org/10.1167/jov.24.6.1 

Below is a poster that summarizes the main results

<img width="857" alt="image" src="https://github.com/Huiyuan-Miao/V1-Nonlinear/assets/126112893/b10086d8-0a6d-4890-80bd-bbce392065b9">

To run the code 
The CNNs (modified AlexNet and VGG19) is in alexnet.py and vgg.py. To train CNNs, use train_CNN.py.   

After training the CNN models, use processFeatureMap.py to extract layer-wise activations.   

To extract Gabor-based features, see createGabor folder.   

To fit V1 model after features are extracted, see v1Model feature, use the exampleAlexnetTrain.py to train V1 models (change feature directory and output directory for different CNN models). 
