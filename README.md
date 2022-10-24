# L1-cSVD
In this repo, an algorithm that performs L1-norm based compact Singular Value Decomposition (L1-cSVD) is implemented. The algorithm aims to robustly estimate singular values from a matrix corrupted by noise and gross and sparse outliers.

The decomposition is written to be

![equation](https://latex.codecogs.com/svg.image?\mathbf{X}\approx\mathbf{U}_{L1}\mathbf{\Sigma}_{L1}\mathbf{V}_{L1}^T&space;)
---
where ![equation]\mathbf{U}_{L1}) and ![equation](https://latex.codecogs.com/svg.image?\mathbf{V}_{L1}) are orthonormal, and ![equation](https://latex.codecogs.com/svg.image?\mathbf{\Sigma}_{L1}) is diagonal. 

![equation](https://latex.codecogs.com/svg.image?\mathbf{U}_{L1}) is taken from L1-PCA. The L1-singular values and right singular vectors are found by solving the following L1-reorthogonalization problem

![equation](https://latex.codecogs.com/svg.image?(\mathbf{\Sigma}_{L1},\mathbf{V}_{L1})=\underset{\mathbf{V}\in\mathbb{S}^{N\times{K}},\mathbf{\Sigma}\in{\rm{diag}}(\mathbb{R}^K)}{\rm{argmin}}||\mathbf{X}^T\mathbf{U}_{L1}-\mathbf{\Sigma}\mathbf{V}^T||_{1,1}&space;)
---

* IEEEXplore article: https://ieeexplore.ieee.org/document/9048980

