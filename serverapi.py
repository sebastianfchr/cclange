from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import roberta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows all origins, but you can restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


MAX_LEN = 96

# "Standard" model with default args 
rob = roberta.RobertaPredictor(MAX_LEN, 'v0-roberta-0.h5') 

class SentenceItem(BaseModel):
    sentence: str
    sentiment: str

class SentenceBatch(BaseModel):
    sentences: list[str]
    sentiments: list[str]

# receive and process a SentenceItem
@app.post("/predict_sentence/")
async def receive_post(item: SentenceItem):
    try:
        sentence_excerpt = rob.predict_sentence(item.sentence, item.sentiment)
        return {"message": "OK", "data": sentence_excerpt}
    except AssertionError:
        return {"message": "Format Error: Input should be of format '{' sentence: str, sentiment: str '}', where sentiment is 'positive'/'neutral'/'negative'", "data": None}
    except:
        return {"message": "Internal Server Error", "data": None}

# receive and process a SentenceBatch
@app.post("/predict_sentence_batch/")
async def receive_post(item: SentenceBatch):
    try:
        sentence_excerpts = rob.predict_sentence_batch(item.sentences, item.sentiments)
        return {"message": "OK", "data": sentence_excerpts}
    except AssertionError:
        return {"message": "Format Error: Input should be of format '{' sentences: list[str], sentiments: list[str] '}', where sentiment is 'positive'/'neutral'/'negative', and len(sentiments)==len(sentences)", "data": None}
    except:
        return {"message": "Internal Server Error", "data": None}


@app.get("/")
def read_root():
    return {"message": "Hello, world!"}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)




# Example requests:

# single on /predict_sentence/ 
# curl -X POST http://localhost:8001/predict_sentence/ -H 'Content-Type: application/json' -d '{"sentence": "Going down the beautiful road, I met a horrible rabbit", "sentiment": "negative"}'
# batch on /predict_sentence_batch/
# curl -X POST http://localhost:8001/predict_sentence_batch/ -H 'Content-Type: application/json' -d '{"sentences": ["Going down the beautiful road, I met a horrible rabbit", "while drinking a craft beer, I became damn hungry"], "sentiments": ["negative", "neutral"]}'

# maps 80 of host to 8080 on host 
# 

# docker run --gpus=all -v $(pwd):/code -p 8001:8001 -w /code -it sebastianfchr/appl_tfdocker:latest -- uvicorn serverapi:app --host 0.0.0.0 --port 8001
