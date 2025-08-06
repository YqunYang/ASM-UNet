# 🧠 ASM-UNet

**Paper**: *ASM-UNet: Adaptive Scan Mamba Integrating Group Commonalities and Individual Variations for Fine-Grained Segmentation*  
**Paper Link**: *[Coming Soon]*  
**Dataset Link**: *[Coming Soon]*

---

## 📦 Installation

Follow the steps below to set up the environment for **ASM-UNet**:

```bash
# Step 0: Clone the repository
git clone https://github.com/YqunYang/ASM-UNet.git
cd ASM-UNet

# Step 1: Create and activate the conda environment
conda create -n ASM_UNet python=3.9.19 -y
conda activate ASM_UNet

# Step 2: Install PyTorch (CUDA 11.8 version; change if needed)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu118

# (Optional) Step 2.1: Install nvcc if not available
conda install -c conda-forge cudatoolkit-dev

# Step 3: Install required Python packages
pip install ninja==1.11.1.1 transformers==4.44.0 einops==0.8.0 packaging

# Step 4: Compile and install the causal-conv1d module
cd causal-conv1d
python setup.py install

# Step 5: Compile and install the mamba module
cd ../mamba
python setup.py install

# Step 6: Install asmunet module
cd ../asmunet
pip install -e .
```


## 📖 Citation

If you find this work useful, please cite:

```
@article{xxx,
  title     = {ASM-UNet: Adaptive Scan Mamba Integrating Group Commonalities and Individual Variations for Fine-Grained Segmentation},
  author    = {Yuqun Yang et al.},
  journal   = {xxx},
  year      = {2025}
}
