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
    model_path = 'model-downloads/opt-6.7b'
    tokenizer_path = 'facebook/opt-6.7b'
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


<<<<<<< HEAD
def generate(max_output_length=256):
    model_path = 'model-downloads/opt-6.7b'
    tokenizer_path = 'facebook/opt-6.7b'
=======
def generate(min_output_length=256, max_output_length=512):
    model_path = 'model-downloads/opt-350m'
    tokenizer_path = 'facebook/opt-350m'
>>>>>>> f2daca1b4fa86ef7be8805c868c6c6461c4a00af
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
        min_length=min_output_length,
        max_length=max_output_length,
        num_beams=4,
        no_repeat_ngram_size=4,
        early_stopping=True,
<<<<<<< HEAD
        temperature=1,
	repetition_penalty=3.0,
	num_beam_groups=2,
	diversity_penalty=0.1
=======
        num_beam_groups=2,
        diversity_penalty=0.1,
        repetition_penalty=3.0,
>>>>>>> f2daca1b4fa86ef7be8805c868c6c6461c4a00af
    )
    
    output_text = tokenizer.decode(beam_output[0], skip_special_tokens=True)
    
    x = tokenizer(output_text)
    print("Num output tokens: " + str(len(x['input_ids'])))

    with open(output_path, 'w') as fhand:
        fhand.write(output_text)

    print("Output text written to " + str(output_path))

if __name__ == '__main__':
    generate()
    
