import fastapi
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import roberta

app = FastAPI()

# Cors must be off
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


MAX_LEN = 96

# "Standard" model with default args 
rob = roberta.RobertaPredictor(MAX_LEN, 'v0-roberta-0.h5') 

class SentenceBatch(BaseModel):
    sentences: List[str]
    sentiments: List[str]


# receive and process a SentenceBatch
@app.post("/predict_sentence_batch/")
async def receive_post(item: SentenceBatch):
    
    if not all([s in ["positive", "negative", "neutral"] for s in item.sentiments]):
        return {"message": "Format Error", "data": "Input should be of format '{' sentences: List[str], sentiments: List[str] '}', where sentiment is 'positive'/'neutral'/'negative', and len(sentiments)==len(sentences)"}
    if (len(item.sentences) != len(item.sentiments)) or len(item.sentences) == 0:
            raise fastapi.HTTPException(status_code=422, detail="Format Error: Input should be of format '{' sentences: List[str], sentiments: List[str] '}', where sentiment is 'positive'/'neutral'/'negative', and len(sentiments)==len(sentences)")
    try:
        sentence_excerpts = rob.predict_sentence_batch(item.sentences, item.sentiments)
        return {"message": "OK", "data": sentence_excerpts}
    except:
        raise fastapi.HTTPException(status_code=500, detail="Internal Server Error")




# Example requests:

# batch on /predict_sentence_batch/
# curl -X POST http://localhost:8123/predict_sentence_batch/ -H 'Content-Type: application/json' -d '{"sentences": ["Going down the beautiful road, I met a horrible rabbit", "while drinking a craft beer, I became damn hungry"], "sentiments": ["negative", "neutral"]}'

# docker run --gpus=all -v $(pwd):/code -p 8123:8123 -w /code -it sebastianfchr/appl_tfdocker:latest -- uvicorn serverapi:app --host 0.0.0.0 --port 8123
