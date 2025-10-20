from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from transformers import pipeline

class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)

class PredictOut(BaseModel):
    label: str
    score: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.clf = pipeline("sentiment-analysis",
                             model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                             revision="714eb0f")
    yield

app = FastAPI(
    title="AI Sentiment & Emotion Classifier",
    description="Classifies text sentiment using a Hugging Face model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz", tags=["Health"])
def health_check():
    return {"status": "ok"}

@app.get("/predict", include_in_schema=False)
def predict_ui():
    # typing /predict in the browser will land on the UI
    return RedirectResponse(url="/frontend/predict.html")

@app.post("/predict", response_model=PredictOut, tags=["Model"])
async def predict(payload: PredictIn, request: Request):
    out = request.app.state.clf(payload.text)[0]
    return {"label": out["label"], "score": float(out["score"])}

# Redirect root "/" to the Home page
@app.get("/", include_in_schema=False)
def redirect_to_home():
    return RedirectResponse(url="/frontend/index.html")

# Serve static frontend (home + predict)
app.mount("/frontend", StaticFiles(directory="app/frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
