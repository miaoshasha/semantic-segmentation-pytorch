#!/usr/bin/env bash
# update gcloud
gcloud components update
REGION=us-central1
# set up cloud bucket for source code
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
SOURCE_BUCKET_NAME=${PROJECT_ID}-mlengine-pytorch_trial
gsutil mb -l $REGION gs://$SOURCE_BUCKET_NAME

# package code
python setup.py sdist

# upload packaged source code
local_file_path="/Users/yishasun/Documents/dl_models/sem_seg/semantic-segmentation-pytorch/dist"
file_name="trainer-0.1.tar.gz"
SOURCE_CODE="gs://$SOURCE_BUCKET_NAME/project/$file_name"
gsutil cp $local_file_path/$file_name $SOURCE_CODE
#uploading to cloud to keep track

# train log bucket
BUCKET_NAME_SUR=${PROJECT_ID}-pytorch_trial
echo $BUCKET_NAME_SUR


PROJECT="project"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="scenenet_$now"

# create the new bucket
out_dir="$BUCKET_NAME_SUR-log"
gsutil mb -l $REGION gs://$out_dir # comment out after first run

# folder for each training
OUTPUT_DIR="gs://$out_dir/$JOB_NAME/logdir"

# bucket for staging
PACKAGE_STAGING_BUCKET="gs://$BUCKET_NAME_SUR-staging-$JOB_NAME"

# create the new bucket
gsutil mb -l $REGION $PACKAGE_STAGING_BUCKET
# location of source project
TRAINER_PACKAGE_PATH="trainer"  # ><<<<<<<<-=========problem here!!!!

# data dir
ADE20K="ade20k"     # <<<<<<<<<<<=========change here!!!

# ckpt
CKPT="ckpt_ade20k"     #<<<<<<<<<<<<<<==========

# custom dependency bucket
#DEPENDENCY_SLIM="/Users/yishasun/Documents/dl_models/models/research/slim/dist/slim-0.1.tar.gz"


#pip install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl
#pip install torchvision 

# hardware to use
TIER="BASIC"

MODEL_PATH="./baseline-resnet34_dilated8-psp_bilinear"

gcloud ml-engine jobs submit training $JOB_NAME \
    --stream-logs \
    --staging-bucket $PACKAGE_STAGING_BUCKET \
    --job-dir "${OUTPUT_DIR}"  \
    --scale-tier "${TIER}"  \
    --module-name trainer.test \
    --region "${REGION}" \
    --packages "${SOURCE_CODE}" \
    -- \
    --model_path="${MODEL_PATH}" \
    --test_img="./davidgarrett.jpg" \
    --arch_encoder="resnet34_dilated8" \
    --arch_decoder="psp_bilinear" \
    --fc_dim=2048 \
    --result="${OUTPUT_DIR}"



