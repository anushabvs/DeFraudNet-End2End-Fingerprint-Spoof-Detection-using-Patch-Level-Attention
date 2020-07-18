# DeFraudNet-End2End-Fingerprint-Spoof-Detection-using-Patch-Level-Attention
This repository contains Pytorch implementation of [DeFraudNet:End2End Fingerprint Spoof Detection using Patch Level Attention](https://ieeexplore.ieee.org/iel7/9087828/9093261/09093397.pdf).

Fingerprint spoofing is a technique in which fake fingerprints are created or obtained illegally to circumvent devices secured using these fingerprints. This feild has seen exemplary improvements in the recent past and is still improving as the spoofing system is also becoming advanced. Various methods have been developed which obtain state-of-the-art results on spoof detection when tested on same material fingerprints. DeFraudNet obtains state-of-the-art results in not only fingerprint spoof detection when tested on the same material subjects but also obtains state-of-the-art cross-sensor, cross-material, cross-dataset results. 

## DeFraudNet Network Overview

DefraudNet network mainly uses DenseNets as the base networks. The complete overview of the network can be seen in the figure below.
 
![alt text](https://github.com/anushabvs/DeFraudNet-End2End-Fingerprint-Spoof-Detection-using-Patch-Level-Attention/blob/master/Images/Defraudnet_network.png "Figure.1. DeFraudNet complete network")

Defraudnet first takes in the raw fingerprint as a input. This fingerprint is pre-processed using Gabor filters and LBP to bring out additional textural features. After pre-rpocessing, the data is passed on for patch extraction where *n* patches are extracted for each fingerprint. This is followed by simultaneous training of two DenseNets for whole image and patch-based feature extractions. After the feature map extraction, channel, spatial and patch attention are performed on the patches. This is followed by the final step of patch feature fusion with the whole image and classification. 
