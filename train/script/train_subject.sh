# *[Specify the config file path and the GPU devices to use]
export CUDA_VISIBLE_DEVICES=0

# *[Specify the config file path]
export OMINI_CONFIG=./train/config/subject.yaml

export WANDB_API_KEY='a5ebf533c17c677bcee36f66c91907b5fb102f7c'

echo $OMINI_CONFIG
export TOKENIZERS_PARALLELISM=true

accelerate launch --main_process_port 41353 -m omini.train_flux.train_subject