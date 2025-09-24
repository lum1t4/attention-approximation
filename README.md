


TODOs:

- distill_individual_layers.py
    - Check why get OOM when in DDP
    - Uniform loss, logging code with whole model distillation script
    - Add tracker (e.g. wandb) to plot loss curves and other stuff
    - On [RADLADS](https://arxiv.org/pdf/2505.03005) layer distillation was done for about 100M tokens with lr cosine from 1e-3 to 1e-5 with bs=32
    - To train 100M tokens if bs=16, seq_len=512 and grad_acc=4 then 100M/(16*512*4)= ~3052 steps
    - Need to check convergence of loss with different rank
    - Maybe attempt to do some generation in this script to have a vague idea of how well the model is doing

- distill_whole_model.py
    - On [RADLADS](https://arxiv.org/pdf/2505.03005) whole model distillation was done for about 500M tokens with lr from 1e-3 to 1e-5.
    - Add tracker (e.g. wandb) to plot loss curves and other stuff
    - The model should be trained with 100M tokens as done in  [RADL


- Run benchmarks with the distilled models: (lambada↑ MMLU↑ arc_c↑ arc_e↑ hella↑ piqa↑ winog)

