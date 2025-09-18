
.PHONY: install
install:
	mkdir -p checkpoints
	mkdir -p data
	curl -L -o data/MobileLLM/model.safetensors https://huggingface.co/mia-llm/MobileLLM-125M-wikitext2raw-hosein/resolve/main/model.safetensors
	curl -L -o data/MobileLLM/config.json https://huggingface.co/mia-llm/MobileLLM-125M-wikitext2raw-hosein/resolve/main/config.json
	pip install "blobfile>=3.0.0" "datasets>=4.0.0" "google>=3.0.0" "ipykernel>=6.30.1" "ipywidgets>=8.1.7" "lm-eval>=0.4.9" "matplotlib>=3.10.5" "nbformat>=5.10.4" "protobuf>=6.32.0" "safetensors>=0.6.2" "sentencepiece>=0.2.1" "tiktoken>=0.11.0" "torchvision>=0.23.0" "transformers>=4.56.0"
	pip install -e .