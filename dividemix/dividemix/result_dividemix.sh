noise_type="aggre rand1 worst"
lambda_u="5 10 15 25 20 30 40 50"
GPUID=0
echo 'Result' > result.log
for ilambda in $lambda_u;
do
for NOISE in $noise_type;
do
    echo "Running noise $NOISE"
    cat  ../../results/cifar10/c10_dividemix$NOISE\_$ilambda.log | grep Accuracy > tmp.log 
    var=$(tail -n 1 tmp.log)  
    echo  [c10] $NOISE\_$ilambda $var >> result.log
done
    cat  ../../results/cifar100/c100_dividemix$NOISE\_$ilambda.log | grep Accuracy > tmp.log 
    var=$(tail -n 1 tmp.log)  
    echo  [c100] $NOISE\_$ilambda $var >> result.log
done

