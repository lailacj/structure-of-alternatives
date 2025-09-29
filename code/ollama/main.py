import ollama

def print_model_response(model, query):
    response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': query,
        },
    ])

    print(response['message']['content'])

if __name__ == '__main__':
    models = ['llama3.1', 'llama3.2', 'gemma2']
    queries =['Write a function to compute Fibonacci numbers using iteration in Rust.', 
              'What is the cube root of 1860867?', 
              'What is the origin of Unix epoch time?',
              'Are AI bots going to turn humans into paperclips? Yes or no?',
              'Question: what kind of bear is best?']

    for model in models:
        for query in queries:
            print(f"Here is the output from: {model.upper()}")
            print_model_response(model,query)
