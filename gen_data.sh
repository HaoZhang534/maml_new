GPU=1
way=5
shot=5
metatrain_iterations=50000
meta_batch_size=5
dataset=omniglot
meta_lr=0.01
update_lr=0.5
poison_lr=0.1
num_updates=10
reptile=True
train=True
mode='train_label_flip'
resume=False
save_poison_ep=50
median=False
noise_rate=0.2
if [ $mode == 'train_with_poison' ]
then
  logdir=logs/${dataset}${way}way${shot}shot/train_poisonreptile/
else
  logdir=logs/${dataset}${way}way${shot}shot/${mode}reptile/
fi
echo $logdir
poison_itr=48700
poison_dir=../poison_reptile_2/logs/omniglot5way5shot/train_poisonreptile/cls_5.mbs_5.ubs_5.numstep10.updatelr0.5.poison_lr0.1batchnorm/poisonx_50.npy
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
CUDA_VISIBLE_DEVICES=$GPU python main_gen_data.py --datasource=$dataset --metatrain_iterations=$metatrain_iterations --meta_batch_size=$meta_batch_size --update_batch_size=$shot --update_lr=$update_lr --num_updates=$num_updates --logdir=$logdir  --reptile=$reptile --mode=$mode --train=$train --resume=$resume --poison_itr=$poison_itr --poison_lr=$poison_lr --meta_lr=$meta_lr --save_poison_ep=$save_poison_ep --median=$median --poison_dir=$poison_dir --noise_rate=$noise_rate
