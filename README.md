# DBEFuse
DBEFuse: Enhancing Infrared and Visible Image Fusion via Dual-Branch Collaborative Enhancement

## Citation

```
If you use this code, please cite our paper:  

@article{zhang2025DBEFuse,
  title={DBEFuse: Enhancing Infrared and Visible Image Fusion via Dual-Branch Collaborative Enhancement},
  journal={The Visual Computer}
}


```



## Abstract

Infrared and visible image fusion aims to combine thermal radiation information from infrared images with texture details from visible images, enabling comprehensive scene perception in applications such as night surveillance and autonomous driving. However, existing methods often suffer from insufficient interaction between global and local representations and a lack of cross-modal correlation modeling, resulting in structural distortion and visual artifacts.This paper proposes DBEFuse, a Dual-Branch Enhancement-based Fusion Network for infrared and visible image fusion. The network employs a Global Structure Extraction Module (GSEM) based on ConvNeXt Blocks to capture long-range dependencies and maintain structural consistency, and a Local Detail Extraction Module (LDEM) with Triplet Attention to enhance fine-grained textures. A six-stream multi-scale feature fusion mechanism is designed to achieve efficient cross-modal information aggregation.A joint loss function integrating structural similarity, mean squared error, gradient, and correlation regularization terms guides the end-to-end optimization of feature decomposition and fusion. Experiments on the TNO, RoadScene, and MSRS datasets demonstrate that DBEFuse achieves superior performance in both objective metrics (EN, MI, Qabf) and visual quality, effectively preserving infrared targets and visible textures with natural contrast and minimal artifacts.

## 🌐 Usage

### ⚙ Network Architecture

Our DBEFuse is implemented in Net.py.   


### 🏊 Training
**1. Virtual Environment**
```
- Python 3.8
- PyTorch 2.0.0
- CUDA 11.8
- 其他依赖：numpy==1.24.3, opencv-python==4.8.0, scikit-image==0.21.0

# install DBEFuse requirements
pip install -r requirements.txt
```

**2. Data Preparation**

Download the TNO or RoadScene dataset and place it in the folder structure:
```
./datasets/
├── train/
│   ├── infrared/
│   └── visible/
└── test/
    ├── infrared/
    └── visible/
```

**3. Pre-Processing**

Run 
```
python preprocessing.py
``` 

**4. DBEFuse Training**

Start training with:
```
python new_train.py
``` 
and the trained model is available in ``'./models/'``.

### 🏄 Testing

Pretrained models

**2. Test datasets**

The test datasets used in the paper have been stored in ``'./test_img/TNO'`` and ``'./test_img/MSRS'``.

### 📊 Evaluation
Quantitative metrics:
SSIM
MI
FMI
SF
Qualitative results:
```
================================================================================
The test result of TNO :
                    EN      SF       MI       SCD     VIF     Qbaf
TSTBFuse           7.14    13.41     2.84     1.45     0.91    0.54
================================================================================

================================================================================
The test result of MSRS :
                    EN      SF       MI       SCD     VIF     Qbaf
TSTBFuse           6.84    12.54    3.35      1.44    1.05    0.67
================================================================================
```

## 🙌 DBEFuse



## 📧 Contact

For questions, contact: 332416020952@zzuli.edu.cn





