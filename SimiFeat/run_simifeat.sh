noise_type="aggre rand1 worst"
lambda_u=5 # 20 30 40 50    5 10 15 25
GPUID=0
method="mv"

for iMETHOD in $method;
do
    lambda_u=5
    for NOISE in $noise_type;
    do
        echo "Running noise $NOISE"
        CUDA_VISIBLE_DEVICES=$GPUID nohup python3 main_fast.py --dataset cifar10 --noise_type $NOISE  --k 10 --pre_type dividemix  --num_epoch 21 --Tii_offset 1.0 --method $iMETHOD --label_file_path ../data/CIFAR-10_human.pt  --lambda_u $lambda_u >  ../results/cifar10/$iMETHOD\_c10_simifeat$NOISE\_$lambda_u.log &
        GPUID=$(($GPUID+1)) 
    done

    lambda_u=30
    CUDA_VISIBLE_DEVICES=$GPUID nohup python3 main_fast.py --dataset cifar100 --noise_type noisy100  --k 10 --pre_type dividemix  --num_epoch 21 --Tii_offset 1.0 --method $iMETHOD --label_file_path ../data/CIFAR-100_human.pt  --lambda_u $lambda_u --num_classes 100 >  ../results/cifar100/$iMETHOD\_c100_simifeat$NOISE\_$lambda_u.log &
    GPUID=$(($GPUID+1)) 
done

