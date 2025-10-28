#!/bin/bash
# =============================================================================
# CoMet: Training Script for MVTec AD
# Paper: Towards Real Unsupervised Anomaly Detection Via Confident Meta-Learning
# ICCV 2025 | https://openaccess.thecvf.com/content/ICCV2025/html/Aqeel_Towards_Real_Unsupervised_Anomaly_Detection_Via_Confident_Meta-Learning_ICCV_2025_paper.html
# =============================================================================

# -------------------------------
# USER CONFIGURATION (EDIT THESE)
# -------------------------------

# Path to MVTec AD dataset (update this!)
DATAPATH="./data/mvtec"

# Sub-datasets to train on (all 15 categories)
DATASETS=('screw' 'pill' 'capsule' 'carpet' 'grid' 'tile' 'wood' 'zipper' 'cable' 'toothbrush' 'transistor' 'metal_nut' 'bottle' 'hazelnut' 'leather')

# Results directory
RESULTS_PATH="results"

# GPU ID (set to -1 for CPU)
GPU_ID=0

# -------------------------------
# DO NOT EDIT BELOW THIS LINE
# -------------------------------

# Build dataset flags: -d screw -d pill ...
DATASET_FLAGS=()
for dataset in "${DATASETS[@]}"; do
    DATASET_FLAGS+=("-d" "$dataset")
done

# Create results directory
mkdir -p "$RESULTS_PATH"

echo "Starting CoMet training on MVTec AD..."
echo "Dataset path: $DATAPATH"
echo "Categories: ${DATASETS[*]}"
echo "Saving results to: $RESULTS_PATH"

python3 main.py \
    --gpu "$GPU_ID" \
    --seed 0 \
    --log_group "comet_mvtec" \
    --log_project "MVTecAD_Results" \
    --results_path "$RESULTS_PATH" \
    --run_name "run_$(date +%Y%m%d_%H%M%S)" \
    net \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 40 \
    --embedding_size 256 \
    --gan_epochs 4 \
    --noise_std 0.015 \
    --dsc_hidden 1024 \
    --dsc_layers 2 \
    --dsc_margin .5 \
    --pre_proj 1 \
    dataset \
    --batch_size 8 \
    --resize 329 \
    --imagesize 288 \
    "${DATASET_FLAGS[@]}" \
    mvtec "$DATAPATH"

echo "Training completed. Results saved in: $RESULTS_PATH"