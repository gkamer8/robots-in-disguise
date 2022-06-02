import os
import openai

API_KEY_FILE_PATH = 'openai_api_key.txt'

def confirm_api_call():
    confirmed = input("Enter 'y' to confirm a call to GPT-3 API: ")
    if confirmed != 'y':
        print("Not confirmed; exiting program.")
        exit(0)

if __name__ == '__main__':

    with open(API_KEY_FILE_PATH, 'r') as fhand:
        openai.api_key = fhand.read()

    output_path = os.path.join('gen-test-files', 'testing.txt')
    context_path = os.path.join('gen-test-files', 'context.txt')

    with open(context_path, 'r') as fhand:
        context = fhand.read()

    confirm_api_call()
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=context,
        temperature=1,
        max_tokens=1024,
        frequency_penalty=.3,
        presence_penalty=.3
    )

    output_text = response['choices'][0]['text']
    output_text = context + output_text  # Prepend context to output

    with open(output_path, 'w') as fhand:
        fhand.write(output_text)

    print("Output text written to " + str(output_path))
