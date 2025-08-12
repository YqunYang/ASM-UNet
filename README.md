# 🧠 ASM-UNet

**Paper**: *ASM-UNet: Adaptive Scan Mamba Integrating Group Commonalities and Individual Variations for Fine-Grained Segmentation*  
**Paper Link**: *[https://arxiv.org/pdf/2508.07237](https://arxiv.org/pdf/2508.07237)*  
**Dataset Link**: *[Available after paper acceptance]*  
**Version Note**: *This is the tested version of the code, which runs normally but differs slightly from the final release. The final version will be made available once the paper is accepted.*

---

## ⚙️ 1. Installation

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
pip install ninja==1.11.1.1 transformers==4.44.0 einops==0.8.0 acvl-utils==0.2 packaging

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

## 📁 2. Data Preparation
```bash
# Step 0: Download and unzip the BTMS dataset
wget [Dataset Link]
unzip Dataset001_BTMS.zip

# Step 1: Move the dataset to nnUNet_raw
mv Dataset001_BTMS/ ASM-UNet/data/nnUNet_raw/

# Step 2: Preprocess using nnUNetv2
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

## 🚀 3. Model Training
```bash
# Train with single-GPU
nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainer_asmunet

# Train with multi-GPUs (example: 2 GPUs)
CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train 1 3d_fullres all -tr nnUNetTrainer_asmunet -num_gpus 2
```

#### (Optional) Solution for Limited GPU Memory
In the file `nnUNetTrainer_asmunet.py`, we provide an optional parameter `if_lmls` to handle cases where GPU memory is insufficient.
When `if_lmls` is set to True, GPU memory usage will be reduced at the cost of increased runtime.
```python
    def build_network_architecture(
        ...
        if_lmls: bool = True
    )
```

## 🖥️ 4. Inference and Evaluation
**Method 1**: You can perform inference and compute Dice scores using the Jupyter notebook:  
📓 `Pred_and_Eval_ASM_UNet.ipynb`
Within the notebook, the following parameters can be configured for evaluation:
```python
Dataset_ID = 1                           # ID of the dataset
fold = "all"                             # Use "all" or a specific fold (e.g., "1")
use_gpu = "2"                            # GPU ID to be used
tr = "nnUNetTrainer_asmunet"             # Trainer name
checkpoint_name = "checkpoint_best.pth"  # Or "checkpoint_latest.pth"
predicited_set = "imagesTs"              # Dataset to predict (typically "imagesTs")
```

**Method 2**: Alternatively, you can run the 📓 Pred_and_Eval_ASM_UNet.py file to accomplish the same task:
```bash
python Pred_and_Eval_ASM_UNet.py \
    --Dataset_ID 1 \
    --fold all \
    --use_gpu 0 \
    --tr nnUNetTrainer_asmunet \
    --checkpoint_name checkpoint_best.pth \
    --predicited_set imagesTs
```


## 📌 Notes
1. Replace 1 with your actual dataset ID if different.
2. For large datasets, use multi-GPU training for faster convergence.
3. By default, the paths for `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` are located under: `ASM-UNet/data/`.
To customize these paths, you can modify the following file: `ASM-UNet/asmunet/nnunetv2/paths.py`. In this file,
the original `nnUNet` configuration using *environment variables* has been *commented out* for convenience and flexibility.

## ⚠️ Warnings
1. Autocast Deprecation Warning: `/torch/utils/checkpoint.py: FutureWarning: torch.cpu.amp.autocast(args...)is deprecated.
Please use torch.amp.autocast('cpu', args...)instead.`
Solution: Update the calls in checkpoint.py (verified working fix) or safely ignore (backwards compatible).
2. Symbolic Shapes Initialization Warnings: rank1]:W0807 ... symbolic_shapes.py:4449] [0/0] xindex is not in var_ranges, defaulting to unknown range.
Safely ignore - only appears during first epoch compilation

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{xxx,
  title     = {ASM-UNet: Adaptive Scan Mamba Integrating Group Commonalities and 
               Individual Variations for Fine-Grained Segmentation},
  author    = {Yuqun Yang et al.},
  journal   = {xxx},
  year      = {2025}
}
```
## 🎉 Acknowledgements
We would like to express our sincere thanks to the following open-source projects that made this work possible:

-  [**nnUNet**](https://github.com/MIC-DKFZ/nnUNet/tree/master): A robust framework for medical image segmentation.
-  [**Mamba**](https://github.com/state-spaces/mamba): State Space Models for efficient sequence modeling.
-  [**causal-conv1d**](https://github.com/Dao-AILab/causal-conv1d): Efficient implementation of causal 1D convolutions.

Their contributions to the research community are truly appreciated.

