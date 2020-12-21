# Skin Cancer MNIST: HAM10000

The Skin MNIST dataset is a classification problem composed of dermatological images allusive to different types of skin cancer.  
Unlike other datasets, this one promotes the study of several types of skin lesions, thus allowing a less generalist diagnosis, and allows a more incisive study on the several types of skin lesions that a patient might suffer.

## Data
This benchmark consists of 10015 images that are the result of an intensive study developed by various entities. The samples are represented in RGB format and have dimensions 600*450 (length and width respectively).  
This benchmark promotes the study of seven different types of skin lesions:  
* Actinic Keratoses;
* Basal cell carcinoma;
* Benign Keratosis;
* Dermatofibroma;
* Melanocytic nevi;
* Melanoma;
* Vascular skin lesions;

## Limitations of this dataset
The main limitations of this benchmarks are:
* High unbalanced classes (Sample distributions between classes are very disproportional);
* Small number of samples;
* Problem with high complexity;
* Samples with high dimensions;

## What this project offers
* Disponibilization of a Jupyter notebook with problem pre-analysis;
* Several techniques are applied to reduce the main limitations of the problem, such as: Random Oversampling, Cost-Sensitive-Learning and Data Augmentation;
* It implements and uses four convolutional architectures for the consequent resolution of the problem: AlexNet, VGGNet, ResNet and DenseNet;
* Use of PSO algorithm to optimize the structure and other hyperparameters of different convolutional architectures;
* Application of the ensemble technique to improve the performance obtained, individually, by the architectures (combining the probabilistic distributions of the different architectures - average);

## Results
The table represented below includes the results related to the optimization of each architecture, and the user can download the consequent model obtained.  
| Model | Memory | Macro Average F1Score | Macro Average Recall | Accuracy | File | 
|---|---|---|---|---|---|
| AlexNet | 7,8 MB | 65.4% | 63.5% | 81.1% | [AlexNet h5 File](https://github.com/bundasmanu/ProjetoMestrado/blob/master/arquiteturas_otimizadas/Skin_MNIST/alexnet_gbest_oficial.h5?raw=true) |
| VGGNet | 12,9 MB | 64.8% | 62.3% | 80.8% | [VGGNet h5 File](https://github.com/bundasmanu/ProjetoMestrado/blob/master/arquiteturas_otimizadas/Skin_MNIST/vggnet_gbest_oficial.h5?raw=true) |
| ResNet | 39,8 MB |  66.5% |  64.2% | 81.3% | [ResNet h5 File](https://github.com/bundasmanu/ProjetoMestrado/blob/master/arquiteturas_otimizadas/Skin_MNIST/resnet_gbest_oficial.h5?raw=true) |
| DenseNet | 4,4 MB | 67.6% |  65.4% | 81.6% | [DenseNet h5 File](https://github.com/bundasmanu/ProjetoMestrado/blob/master/arquiteturas_otimizadas/Skin_MNIST/densenet_gbest_oficial.h5?raw=true) |
| Ensemble Average All Models | 21,8 MB | 68.5% | 65.2%  | 83.0% | [Ensemble All Models h5 File](https://github.com/bundasmanu/ProjetoMestrado/blob/master/arquiteturas_otimizadas/Skin_MNIST/ensemble_all.h5?raw=true) |
| Ensemble Average Alex + VGG + Dense | 17,5 MB | 69.2% | 66.5%  | 83.1% | [Ensemble Best Combination h5 File](https://github.com/bundasmanu/ProjetoMestrado/blob/master/arquiteturas_otimizadas/Skin_MNIST/ensemble_best.h5?raw=true) |

## Data Acess
https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
