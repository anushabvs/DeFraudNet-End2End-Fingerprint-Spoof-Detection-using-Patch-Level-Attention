# DeFraudNet-End2End-Fingerprint-Spoof-Detection-using-Patch-Level-Attention
This repository contains Pytorch implementation of [DeFraudNet:End2End Fingerprint Spoof Detection using Patch Level Attention](https://ieeexplore.ieee.org/iel7/9087828/9093261/09093397.pdf).

Fingerprint spoofing is a technique in which fake fingerprints are created or obtained illegally to circumvent devices secured using these fingerprints. This field has seen exemplary improvements in the recent past and is still improving as the spoofing system is also becoming advanced. Various methods have been developed which obtain state-of-the-art results on spoof detection when tested on same material fingerprints. DeFraudNet obtains state-of-the-art results in not only fingerprint spoof detection when tested on the same material subjects but also obtains state-of-the-art cross-sensor, cross-material, cross-dataset results. 

## DeFraudNet Network Overview

DefraudNet network mainly uses DenseNets as the base networks. The complete overview of the network can be seen in the figure below.
 
![alt text](https://github.com/anushabvs/DeFraudNet-End2End-Fingerprint-Spoof-Detection-using-Patch-Level-Attention/blob/master/Images/Defraudnet_network.png "Figure.1. DeFraudNet complete network")
<p align="center">
 Figure.1. Complete network overview
</p>


Defraudnet first takes in the raw fingerprint as a input. This fingerprint is pre-processed using Gabor filters and LBP to bring out additional textural features. After pre-processing, the data is passed on for patch extraction where *n* patches are extracted for each fingerprint. This is followed by simultaneous training of two DenseNets for whole image as well as patch-based feature extractions. After the feature map extraction, channel, spatial and patch attention are performed on the patches. This is followed by the final step of patch feature fusion with the whole image and classification. This network obtains state-of-the-art cross-sensor , cross-material and cross-dataset results. The complete analysis can be seen in the next section of results and analysis. This is due the innate property of DeFraudNet to learn common characteristics among the datasets which help in the subsequent classification. If we plot the GradCAM of patch attention model we get the results as seen in Figure.2.

<p align="center">
 <img src="https://github.com/anushabvs/DeFraudNet-End2End-Fingerprint-Spoof-Detection-using-Patch-Level-Attention/blob/master/Images/GradCAM_to_patch_attention.png" height="350" width="600">
</p>

<p align="center">
 Figure.2. GradCAM applied on patch attention network
</p>

We can see from Figure.2. that even though the network is not initialized with minutiae points, while training, the model inherently learns and considers regions close to minutiae points as an important feature for spoof detection. These inherent minutiae based features obtained by the network help in obtaining improved cross-sensor, cross-material and cross-dataset results.
## Results and Analysis
This network is trained and tested on four publically available datasets: LivDet 2017, LivDet 2015, LiveDet 2013 and LivDet 2011. Exhaustive experimentation and comparision has been done on these four datasets and the network is tested for it's cross-material, cross-sensor and cross-dataset performance. The comparative results obtained by DeFraudNet can be seen below.
<p align="center">
 <img src="https://github.com/anushabvs/DeFraudNet-End2End-Fingerprint-Spoof-Detection-using-Patch-Level-Attention/blob/master/Images/Same-material_network_performance.png" height="250" width="500">
</p>

<p align="center">
 Table.1. Same-Material Network Performance
</p>

<p align="center">
 <img src="https://github.com/anushabvs/DeFraudNet-End2End-Fingerprint-Spoof-Detection-using-Patch-Level-Attention/blob/master/Images/cross-material_network_performance.png" height="220" width="380">
</p>

<p align="center">
 Table.2. Cross-Material Network Performance
</p>


<p align="center">
 <img src="https://github.com/anushabvs/DeFraudNet-End2End-Fingerprint-Spoof-Detection-using-Patch-Level-Attention/blob/master/Images/cross-sensor_network_performance.png" height="290" width="600">
</p>

<p align="center">
 Table.3. Cross-Sensor Network Performance
</p>

<p align="center">
 <img src="https://github.com/anushabvs/DeFraudNet-End2End-Fingerprint-Spoof-Detection-using-Patch-Level-Attention/blob/master/Images/cross-dataset_network_performance.png" height="220" width="390">
</p>

<p align="center">
 Table.4. Cross-Dataset Network Performance
</p>

