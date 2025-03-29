from fastapi import FastAPI
from pydantic import BaseModel


# import roberta.models
# rob = roberta.models.MyRobertaModel()

import roberta
import tokenizers

tokenizer = tokenizers.ByteLevelBPETokenizer.from_file('roberta/config/vocab-roberta-base.json', 'roberta/config/merges-roberta-base.txt', lowercase=True, add_prefix_space=True) 
roberta.RobertaPredictor(96, 'v0-roberta-0.h5', tokenizer)

app = FastAPI()

class Item(BaseModel):
    sentence: str
    sentiment: str # technically, this could be

@app.post("/submit/")
async def receive_post(item: Item):
    print(item.sentence)
    return {"message": "Received!", "data": item}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)