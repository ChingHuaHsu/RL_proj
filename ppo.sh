python main.py --env-name "Flip_Flop" --algo ppo  --check_point /home/mchiou/miniconda3/a_proj/pytorch-a2c-ppo-acktr-gail/trained_models/ppo/Flip_Flop.pt --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 49 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01