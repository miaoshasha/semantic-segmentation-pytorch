#!/bin/bash -l

start_date=`date +"%Y-%m-%d %T.%6N"`
echo "$start_date #1:Script starts"


source $HOME/Documents/penvs/ENV/bin/activate

#pip install numpy
#python -m pip install scipy
#pip install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl 
#pip install torchvision 


MODEL_PATH=./baseline-resnet34_dilated8-psp_bilinear
RESULT_PATH=./
#mkdir $MODEL_PATH

#wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/baseline-resnet34_dilated8-psp_bilinear/encoder_best.pth
#wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/baseline-resnet34_dilated8-psp_bilinear/decoder_best.pth
#wget -P $RESULT_PATH http://sceneparsing.csail.mit.edu//data/ADEChallengeData2016/images/validation/ADE_val_00000278.jpg

python3 -u test.py \
  --model_path $MODEL_PATH \
  --test_img ./davidgarrett.jpg \
  --arch_encoder resnet34_dilated8 \
  --arch_decoder psp_bilinear \
  --fc_dim 512 \
  --result $RESULT_PATH

deactivate

end_date=`date +"%Y-%m-%d %T.%6N"`
echo "$end_date #1:Script ends"
echo "takes $end_date-$start_date"