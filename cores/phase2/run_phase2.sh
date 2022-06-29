# CUDA_VISIBLE_DEVICES=1,2,3,5 nohup python3 phase2.py -c confs/resnet34_c10_worst_mv.yaml --unsupervised > phase2_c10_worst_mv_inst4para_autoaug_lr001.log &
# CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 nohup python3 phase2.py -c confs/resnet34_c10_rand_mv.yaml --unsupervised > phase2_c10_rand_mv.log &
# CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 nohup python3 phase2.py -c confs/resnet34_c10_aggre_mv.yaml --unsupervised > phase2_c10_aggre_mv.log &
# CUDA_VISIBLE_DEVICES=6 nohup python3 phase2.py -c confs/resnet34_c100_noisy100_mv_para1.yaml --unsupervised > phase2_c100_noisy100_mv_para1.log &
# CUDA_VISIBLE_DEVICES=4 nohup python3 phase2.py -c confs/resnet34_c10_worst_mv_para1.yaml --unsupervised > phase2_c10_worst_mv_para1.log &

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python3 phase2.py -c confs/resnet34_c100_noisy100_mv_para12_balance.yaml --unsupervised > c100_12.log 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python3 phase2.py -c confs/resnet34_c100_noisy100_mv_para13_balance.yaml --unsupervised > c100_13.log 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python3 phase2.py -c confs/resnet34_c100_noisy100_mv_para2_balance.yaml --unsupervised > c100_2.log 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python3 phase2.py -c confs/resnet34_c100_noisy100_mv_para3_balance.yaml --unsupervised > c100_3.log 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 phase2.py -c confs/resnet34_c10_worst_mv_para12.yaml --unsupervised > c10_12.log 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 phase2.py -c confs/resnet34_c10_worst_mv_para13.yaml --unsupervised > c10_13.log 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 phase2.py -c confs/resnet34_c10_worst_mv_para2.yaml --unsupervised > c10_2.log 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 phase2.py -c confs/resnet34_c10_worst_mv_para3.yaml --unsupervised > c10_3.log 


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python3 phase2.py -c confs/resnet34_c100_noisy100_mv_para31_balance.yaml --unsupervised > c100_31_b.log 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python3 phase2.py -c confs/resnet34_c100_noisy100_mv_para32_balance.yaml --unsupervised > c100_32_b.log 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python3 phase2.py -c confs/resnet34_c100_noisy100_mv_para3.yaml --unsupervised > c100_3_ub.log 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python3 phase2.py -c confs/resnet34_c100_noisy100_mv_para31.yaml --unsupervised > c100_31_ub.log 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python3 phase2.py -c confs/resnet34_c100_noisy100_mv_para32.yaml --unsupervised > c100_32_ub.log 


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python3 phase2.py -c confs/resnet34_c10_worst_mv_para31_balance.yaml --unsupervised > c10_31_b.log 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python3 phase2.py -c confs/resnet34_c10_worst_mv_para32_balance.yaml --unsupervised > c10_32_b.log 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python3 phase2.py -c confs/resnet34_c10_worst_mv_para31.yaml --unsupervised > c10_31_ub.log 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python3 phase2.py -c confs/resnet34_c10_worst_mv_para32.yaml --unsupervised > c10_32_ub.log 