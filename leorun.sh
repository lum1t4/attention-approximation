# salloc -A IscrC_LAM-next -p boost_usr_prod --qos=boost_qos_lprod --gres=gpu:4 --mem=0 --time=10:00:00
# salloc -A IscrC_LAM-next -p boost_usr_prod --qos=boost_qos_lprod --gres=gpu:4 --mem=0 --time=3-00:00:00
# srun --pty bash
# cd $FAST/attention-approximation/ && source .venv/bin/activate
torchrun --standalone --nproc_per_node=4 scripts/distill_individual_layers.py --config "config/distll-layers.yml" --tracker "wandb" --factorization_rank 16 --checkpoint_dir "C16"
torchrun --standalone --nproc_per_node=4 scripts/distill_whole_model.py --config "config/distill-whole.yml" --tracker "wandb" --factorization_rank 16 --checkpoint_dir "CF16" --from_checkpoint "C16/last.pt"
torchrun --standalone --nproc_per_node=4 scripts/distill_individual_layers.py --config "config/distll-layers.yml" --tracker "wandb" --factorization_rank 16 --checkpoint_dir "C32"
torchrun --standalone --nproc_per_node=4 scripts/distill_whole_model.py --config "config/distill-whole.yml" --tracker "wandb" --factorization_rank 16 --checkpoint_dir "CF32" --from_checkpoint "C32/last.pt"
torchrun --standalone --nproc_per_node=4 scripts/distill_individual_layers.py --config "config/distll-layers.yml" --tracker "wandb" --factorization_rank 16 --checkpoint_dir "C64"
torchrun --standalone --nproc_per_node=4 scripts/distill_whole_model.py --config "config/distill-whole.yml" --tracker "wandb" --factorization_rank 16 --checkpoint_dir "CF64" --from_checkpoint "C64/last.pt"