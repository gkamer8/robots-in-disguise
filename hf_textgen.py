from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer, OPTForCausalLM, GPT2LMHeadModel
import os

"""

An attempt at using hugging face transformers for text generation.

"""

# list contents of model downloads folder
def list_downloads():
    model_downloads_folder = os.path.join('model-downloads')
    contents = os.listdir(model_downloads_folder)
    text = 'Model Downloads:\n' + '\n'.join(contents)
    print(text)

# Simple text generation using pipeline
def default_generation():
    model_path = 'model-downloads/opt-1.3b'
    tokenizer_path = 'facebook/opt-1.3b'
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


def generate(max_output_length=256):
    model_path = 'model-downloads/opt-1.3b'
    tokenizer_path = 'facebook/opt-1.3b'
    output_path = os.path.join('gen-test-files', 'testing.txt')
    context_path = os.path.join('gen-test-files', 'context.txt')

    print("Loading model from " + model_path)

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    model = OPTForCausalLM.from_pretrained(model_path, ignore_mismatched_sizes=True)

    print("Model loaded.")

    with open(context_path, 'r') as fhand:
        context = fhand.read()

    beam_output = model.generate(
        tokenizer.encode(context, return_tensors='pt'),
        max_length=max_output_length,
        num_beams=5,
        no_repeat_ngram_size=4,
        early_stopping=True,
        temperature=.5
    )
    
    output_text = tokenizer.decode(beam_output[0], skip_special_tokens=True)
    
    x = tokenizer(output_text)
    print("Num output tokens: " + str(len(x['input_ids'])))

    with open(output_path, 'w') as fhand:
        fhand.write(output_text)

    print("Output text written to " + str(output_path))

if __name__ == '__main__':
    generate()
    
