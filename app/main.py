# app/main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from transformers import pipeline

class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)

class PredictOut(BaseModel):
    label: str
    score: float
# 1) Lifespan: where you'd load heavy resources (e.g., ML models) once
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.clf = pipeline("sentiment-analysis",
                             model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                             revision="714eb0f")
    yield
    # e.g., clean up resources if needed

# 2) Create the ASGI app object
app = FastAPI(
    title="AI Sentiment & Emotion Classifier",
    description="A simple API that will classify text emotions using a Hugging Face model.",
    version="1.0.0",
    lifespan=lifespan,  # ensures startup/shutdown events run
)

# 3) CORS middleware (lets your frontend or other tools call this API safely)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten later to your domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4) Basic routes
@app.get("/", tags=["Health"])
def root():
    return {"message": "Welcome to the Sentiment & Emotion Classifier API!_3"}

@app.get("/healthz", tags=["Health"])
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictOut, tags=["Model"])
async def predict(payload: PredictIn, request: Request):
    out = request.app.state.clf(payload.text)[0]
    return {"label": out["label"], "score": float(out["score"])}

# 5) Local dev entrypoint (so you can run: `python -m app.main`)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
