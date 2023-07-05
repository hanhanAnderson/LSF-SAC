# EASY

CUDA_VISIBLE_DEVICES=3 python src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=1c3s5z w=0.5 epsilon_anneal_time=100000 t_max=2005000 
sleep 3

CUDA_VISIBLE_DEVICES=3 python src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=3s5z w=0.5 epsilon_anneal_time=100000 t_max=2005000 
sleep 3

CUDA_VISIBLE_DEVICES=3 python src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=8m w=0.5 epsilon_anneal_time=100000 t_max=2005000 
sleep 3

# MED

CUDA_VISIBLE_DEVICES=3 python src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=3s_vs_5z w=0.5 epsilon_anneal_time=100000 t_max=3005000 
sleep 3

CUDA_VISIBLE_DEVICES=3 python src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=5m_vs_6m w=0.5 epsilon_anneal_time=100000 t_max=4005000 
sleep 3

CUDA_VISIBLE_DEVICES=3 python src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=MMM2 w=0.5 epsilon_anneal_time=100000 t_max=2005000 

sleep 3

#HARD

CUDA_VISIBLE_DEVICES=3 python src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=corridor w=0.5 epsilon_anneal_time=100000 t_max=5005000 
sleep 3

# CUDA_VISIBLE_DEVICES=3 python src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=6h_vs_8z w=0.5 epsilon_anneal_time=500000 t_max=5005000 
# sleep 3

#CUDA_VISIBLE_DEVICES=3 python src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=6h_vs_8z w=0.5 epsilon_anneal_time=500000 t_max=5005000 mean_mul=3
#sleep 3

CUDA_VISIBLE_DEVICES=3 python src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=6h_vs_8z w=0.5 epsilon_anneal_time=500000 t_max=5005000  seed=213480556


CUDA_VISIBLE_DEVICES=3 python src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=27m_vs_30m w=0.5 epsilon_anneal_time=100000 t_max=2005000
#CUDA_VISIBLE_DEVICES=2 python src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=27m_vs_30m w=0.5 epsilon_anneal_time=100000 t_max=2005000 batch_size_run=2  buffer_size=2500 batch_size=32
#sleep 3


