from fastapi import FastAPI
from pydantic import BaseModel


import roberta
import tokenizers


app = FastAPI()

# "Standard" model with default args 
rob = roberta.RobertaPredictor() 

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



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)


# Example requests:

# single on /predict_sentence/ 
# curl -X 'POST' 'http://127.0.0.1:8001/predict_sentence/'      -H 'Content-Type: application/json'      -d '{"sentence": "Going down the beautiful road, I met a horrible rabbit", "sentiment": "negative"}'
# batch on /predict_sentence_batch/
# curl -X 'POST' 'http://127.0.0.1:8001/predict_sentence_batch/'      -H 'Content-Type: application/json'      -d '{"sentences": ["Going down the beautiful road, I met a horrible rabbit", "while drinking a craft beer, I became damn hungry"], "sentiments": ["negative", "neutral"]}'
