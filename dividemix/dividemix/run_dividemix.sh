noise_type="aggre rand1 worst"  #aggre rand1 worst
lambda_u=5 
GPUID=0

for NOISE in $noise_type;
do
    echo "Running noise $NOISE"
    CUDA_VISIBLE_DEVICES=$GPUID nohup python3 Train_cifar.py --noise_mode $NOISE --noise_file ../../data/CIFAR-10_human.pt --num_class 10 --dataset cifar10 --is_human --num_epochs 100 --lambda_u $lambda_u >  ../../results/cifar10/c10_dividemix$NOISE\_$lambda_u.log &
    GPUID=$(($GPUID+1)) 
done

lambda_u=30
CUDA_VISIBLE_DEVICES=$GPUID nohup python3 Train_cifar.py --noise_mode noisy100 --noise_file ../../data/CIFAR-100_human.pt --num_class 100 --dataset cifar100 --is_human --num_epochs 100 --lambda_u $lambda_u >  ../../results/cifar100/c100_dividemix$NOISE\_$lambda_u.log &
