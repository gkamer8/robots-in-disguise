from transformers import AutoModel
import os

# Downloads models from hugging face transformers

# Note: list available at https://huggingface.co/models
# Cached models are by default stored in ~/.cache/huggingface/transformers/

def download_model(model_hf_path, download_folder):
    model = AutoModel.from_pretrained(model_hf_path)
    model.save_pretrained(download_folder)

if __name__ == '__main__':
    MODEL_TO_DOWNLOAD = 'distilgpt2'
    MODEL_DOWNLOAD_FOLDER = 'model-downloads'
    download_folder = os.path.join(MODEL_DOWNLOAD_FOLDER, MODEL_TO_DOWNLOAD)
    download_model(MODEL_TO_DOWNLOAD, download_folder)

