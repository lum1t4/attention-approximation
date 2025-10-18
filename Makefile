
.PHONY: install
install:
	mkdir -p  data/MobileLLM/
	curl -L -o data/MobileLLM/model.safetensors https://huggingface.co/mia-llm/MobileLLM-125M-wikitext2raw-hosein/resolve/main/model.safetensors
	curl -L -o data/MobileLLM/config.json https://huggingface.co/mia-llm/MobileLLM-125M-wikitext2raw-hosein/resolve/main/config.json
	mkdir -p data/minipile
	curl -L -o "data/minipile/dataset-train.bin" "https://landslide-big-data-2425.s3.eu-central-1.amazonaws.com/dataset-train.bin"
	curl -L -o "data/minipile/index-train.bin" "https://landslide-big-data-2425.s3.eu-central-1.amazonaws.com/index-train.bin"
	curl -L -o "data/minipile/dataset-validation.bin" "https://landslide-big-data-2425.s3.eu-central-1.amazonaws.com/dataset-validation.bin"
	curl -L -o "data/minipile/index-validation.bin" "https://landslide-big-data-2425.s3.eu-central-1.amazonaws.com/index-validation.bin"