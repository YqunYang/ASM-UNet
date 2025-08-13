# %% [markdown]
# # 🧠 U-Net Inference Script (Single Dataset)
# This notebook runs inference using a selected dataset with nnU-Net.

# %%
import json
import os
import warnings

import torch
from asmunet.evalute_utils import compute_metrics_on_folder, labels_to_list_of_regions
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_raw, nnUNet_results

# %% [markdown]
# ### 🔧 Inference Configuration
# Set key parameters for nnU-Net inference, including dataset ID, trainer, checkpoint, and GPU usage.

# %%
import argparse

def inf_eval():
    parser = argparse.ArgumentParser(
        description="Prediction and Evaluation Script for ASM-UNet"
    )
    parser.add_argument("--Dataset_ID", type=int, default=1, help="ID of the dataset")
    parser.add_argument(
        "--fold", type=str, default="all", help="Use 'all' or a specific fold (e.g., '1')"
    )
    parser.add_argument("--use_gpu", type=str, default="0", help="GPU ID to be used")
    parser.add_argument(
        "--tr", type=str, default="nnUNetTrainer_asmunet", help="Trainer name"
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        choices=["checkpoint_final.pth", "checkpoint_best.pth", "checkpoint_latest.pth"],
        default="checkpoint_best.pth",
        help="Checkpoint file name",
    )
    parser.add_argument(
        "--predicited_set",
        type=str,
        default="imagesTs",
        help="Dataset to predict (typically 'imagesTs')",
    )
    args = parser.parse_args()
    
    Dataset_ID = args.Dataset_ID
    fold = args.fold
    use_gpu = args.use_gpu
    tr = args.tr
    checkpoint_name = args.checkpoint_name
    predicited_set = args.predicited_set
    
    
    # %% [markdown]
    # ### 🎯 Dataset Selection and GPU Configuration
    # Set the GPU device and identify the target dataset based on the given ID.
    
    # %%
    os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu
    dataset_name = [i for i in os.listdir(nnUNet_raw) if str(Dataset_ID).zfill(3) in i]
    assert len(dataset_name) != 0, f"No dataset found for ID {Dataset_ID:03d}"
    dataset_name = dataset_name[0]
    
    # %% [markdown]
    # ### 🚀 Initialize nnU-Net Predictor
    # Set up the nnU-Net predictor and load the trained model checkpoint.
    
    # %%
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device("cuda", 0),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        predictor.initialize_from_trained_model_folder(
            join(nnUNet_results, dataset_name + "/" + tr + "__nnUNetPlans__3d_fullres"),
            use_folds=(fold),
            checkpoint_name=checkpoint_name,
        )
    
    # %% [markdown]
    # ### 🧠 Run Inference on Test Set
    # Perform prediction using the initialized nnU-Net model on the specified dataset.
    
    # %%
    p = predictor.predict_from_files(
        join(nnUNet_raw, dataset_name, predicited_set),
        join(
            nnUNet_results,
            dataset_name,
            f"{tr}__nnUNetPlans__3d_fullres",
            f"{predicited_set}_predlowres",
            f"fold_{fold}",
        ),
        save_probabilities=False,
        overwrite=False,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )
    
    # %% [markdown]
    # ### 📊 Evaluate Segmentation Results
    # Compute evaluation metrics by comparing predictions with reference labels and save them to a JSON summary.
    
    # %%
    folder_ref = join(nnUNet_raw, dataset_name, "labels" + predicited_set[-2:])
    folder_pred = join(
        nnUNet_results,
        dataset_name,
        f"{tr}__nnUNetPlans__3d_fullres",
        f"{predicited_set}_predlowres",
        f"fold_{fold}",
    )
    output_file = join(
        nnUNet_results,
        dataset_name,
        f"{tr}__nnUNetPlans__3d_fullres",
        f"{predicited_set}_predlowres",
        f"fold_{fold}",
        "summary.json",
    )
    image_reader_writer = SimpleITKIO()
    file_ending = ".nii.gz"
    label_len = len(
        json.load(open(join(nnUNet_raw, dataset_name, "dataset.json")))["labels"]
    )
    regions = labels_to_list_of_regions(list(range(1, label_len)))
    ignore_label = None
    num_processes = 12
    results = compute_metrics_on_folder(
        folder_ref,
        folder_pred,
        output_file,
        image_reader_writer,
        file_ending,
        regions,
        ignore_label,
        num_processes,
    )
    # %%
    r = json.load(open(output_file))
    for label, metrics in r["mean"].items():
        print(f"Label: {label}", f"Dice: {metrics['Dice']}")
    print(r["foreground_mean"]["Dice"])


if __name__ == '__main__':
    inf_eval()