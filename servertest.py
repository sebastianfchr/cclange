from fastapi import FastAPI
from pydantic import BaseModel


import roberta
import tokenizers

# tokenizer = tokenizers.ByteLevelBPETokenizer.from_file('roberta/config/vocab-roberta-base.json', 'roberta/config/merges-roberta-base.txt', lowercase=True, add_prefix_space=True) 
# roberta.RobertaPredictor(96, 'v0-roberta-0.h5', tokenizer)
rob = roberta.RobertaPredictor() # "standard" constructor taking default args

app = FastAPI()

class Item(BaseModel):
    sentence: str
    sentiment: str # technically, this could be

@app.post("/submit/")
async def receive_post(item: Item):
    try:
        sentence_excerpt = rob.predict_sentence(item.sentence, item.sentiment)
        return {"message": "OK", "data": sentence_excerpt}
    except AssertionError:
        return {"message": "Format Error: Input should be of format '{' sentence: str, sentiment: str '}', where sentiment is \"positive\"/\"neutral\"/\"negative\" ", "data": None}
    except:
        return {"message": "Internal Server Error", "data": None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)