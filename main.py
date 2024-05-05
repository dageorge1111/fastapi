# This is a sample Python script.
from starlette.responses import JSONResponse

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ast  # for converting embeddings saved as strings back to arrays
from openai import OpenAI # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
from scipy import spatial  # for calculating vector similarities for search
import tiktoken
import requests
from io import StringIO

import ast  # for converting embeddings saved as strings back to arrays
from openai import OpenAI # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
from scipy import spatial  # for calculating vector similarities for search
import tiktoken
from fastapi import FastAPI
from dotenv import load_dotenv 

load_dotenv() 

app = FastAPI()

api_key_openai = os.environ.get('OpenAI_Key') 

client_OpenAI = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=api_key_openai,
)

GPT_MODEL = "gpt-3.5-turbo"

import pandas as pd
import requests
from io import StringIO

def download_csv_to_dataframe(url: str) -> pd.DataFrame:
    filename = "data.csv"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"File saved as {filename}.")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
    df = pd.read_csv(filename)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
    return(df)

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client_OpenAI.embeddings.create(input=[text], model=model).data[0].embedding

def strings_ranked_by_relatedness(
    query: str,
    df: str,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
):
    df = download_csv_to_dataframe(df)
    """Returns a list of strings and relatednesses, sorted from most related to least."""

    query_embedding = get_embedding(query)
    strings_and_relatednesses = [
        (row["PDF Content"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: str,
    model: str,
    token_budget: int = 4000
) -> str:
    df = download_csv_to_dataframe(df)
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below articles on a course. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\narticle section:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


app = FastAPI()

class QueryInput(BaseModel):
    question: str
    link: str

@app.post("/process-query/")
async def process_query(input_data: QueryInput):
    try:
        # Use the question and link in your query_message function
        result = query_message(input_data.question, input_data.link, "gpt-3.5-turbo")
        return JSONResponse(content={"message": result}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"message": f"Failed during cleanup: {str(e)}"}, status_code=500)