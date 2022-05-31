from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer, OPTForCausalLM
import os

# list contents of model downloads folder
def list_downloads():
    model_downloads_folder = os.path.join('model-downloads')
    contents = os.listdir(model_downloads_folder)
    text = 'Model Downloads:\n' + '\n'.join(contents)
    print(text)

if __name__ == '__main__':

    model_path = 'model-downloads/opt-350m'
    tokenizer_path = 'facebook/opt-350m'
    output_path = os.path.join('gen-test-files', 'testing.txt')
    context_path = os.path.join('gen-test-files', 'context.txt')

    print("Loading model from " + model_path)

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    model = OPTForCausalLM.from_pretrained(model_path, ignore_mismatched_sizes=True)

    print("Model loaded.")

    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=512)
    
    with open(context_path, 'r') as fhand:
        context = fhand.read()
    
    output = generator(context)
    output_text = output[0]['generated_text']
    
    x = tokenizer(output_text)
    print("Num output tokens: " + str(len(x['input_ids'])))

    with open(output_path, 'w') as fhand:
        fhand.write(output_text)

    print("Output text written to " + str(output_path))