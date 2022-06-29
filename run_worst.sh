# aggre
lambda_u=5 
NOISE="worst"
GPUID=2

# pre-train
cd dividemix/dividemix 

CUDA_VISIBLE_DEVICES=$GPUID python3 Train_cifar.py --noise_mode $NOISE --noise_file ../../data/CIFAR-10_human.pt --num_class 10 --dataset cifar10 --is_human --num_epochs 100 --lambda_u $lambda_u >  ../../results/cifar10/c10_dividemix$NOISE\_$lambda_u.log 

# data cleaning
cd ../../SimiFeat 
CUDA_VISIBLE_DEVICES=$GPUID python3 main_fast.py --dataset cifar10 --noise_type $NOISE  --k 10 --pre_type dividemix  --num_epoch 21 --Tii_offset 1.0 --method mv --label_file_path ../data/CIFAR-10_human.pt  --lambda_u $lambda_u >  ../results/cifar10/mv_c10_simifeat$NOISE\_$lambda_u.log 

# semi-supervised learning
cd ../cores/phase2 
CUDA_VISIBLE_DEVICES=$GPUID python3 phase2.py -c confs/resnet34_c10_$NOISE\_mv.yaml --unsupervised > ../../learning_$NOISE.log 

# detection 
cd ../../SimiFeat 
CUDA_VISIBLE_DEVICES=$GPUID python3 simifeat_detection.py --dataset cifar10 --noise_type $NOISE  --k 10 --pre_type cores  --num_epoch 21 --Tii_offset 1.0 --method mv --label_file_path ../data/CIFAR-10_human.pt  --lambda_u $lambda_u >  ../detection_$NOISE.log  