import os
import openai
import json
import pinecone
from pathlib import Path
from pprint import pprint
from dotenv import load_dotenv

load_dotenv()

# init OpenAI
openai.api_key = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-3.5-turbo"
openai.Engine.list()  # check we have authenticated

# init Pinecone
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

index = pinecone.Index(PINECONE_INDEX)

def query_openAI(prompt):
    completion = openai.ChatCompletion.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a medical assistant."},
            {"role": "user", "content": f"{prompt}"}
        ],
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return completion


def insert_data():
    data_dir = './data/'
    files_in_dir = os.listdir(data_dir)
    for file in files_in_dir:
        file_path = os.path.join(data_dir, file)
        print("Processing file", file_path)
        
        data = json.loads(Path(file_path).read_text())
        data = data[:2048] # limit to 2048 questions for now
        batch_size=32

        for d in range(0, len(data), batch_size):
            print("processing batch", int(d/batch_size), "of", int(len(data)/batch_size))

            batch = data[d:d+batch_size]
            batch = [b['answer'] for b in batch]
            if (batch):
                try:
                    res = openai.Embedding.create(
                        input=batch,
                        engine=EMBEDDING_MODEL
                    )
                except Exception as e:
                    print("Error", e, batch)
                    exit()
                embeds = [record['embedding'] for record in res['data']]
                ids_batch = [str(n) for n in range(d, d+batch_size)]

                # prep metadata and upsert batch
                meta = [{'text': line} for line in batch]
                to_upsert = zip(ids_batch, embeds, meta)
                # upsert to Pinecone
                index.upsert(vectors=list(to_upsert))

def query_data():
    user_prompt_text = "Enter your question: "

    print(user_prompt_text) # ask the user for a question

    query = input()

    xq = openai.Embedding.create(input=query, engine=EMBEDDING_MODEL)['data'][0]['embedding']
    res = index.query([xq], top_k=5, include_metadata=True)
    contexts = [c['metadata']['text'] for c in res['matches']]
    # pprint("relevant context", res)

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question ONLY based on the context below. DO NOT MENTION CONSULTING with a doctor or healthcare provider as part of the answer. \n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    prompt = ""

    # append contents until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n --- \n\n".join(contexts[:i]) + prompt_end) >= 3500:
            prompt = (prompt_start + "\n\n --- \n\n".join(contexts[:i-1]) + prompt_end)
            break
        elif i == len(contexts) - 1:
            prompt = (prompt_start + "\n\n --- \n\n".join(contexts) + prompt_end)
            break

    # print("prompt", prompt)
    return prompt


def main():
    # insert_data()
    openai_response = query_openAI(query_data())
    print(openai_response.choices[0].message.content)

main()