import librosa
import torch
import gradio as gr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile

# Initialisation de FastAPI
app = FastAPI(
    title="API de Détection de Sentiment dans les Audios",
    description="Transcription avec Wav2Vec2 + Analyse de sentiment avec BERT",
    version="1.0"
)

# chargements d'Algorithme de wav2vec2.0

print("Loading Wav2Vec2 (speech-to-text)...")
asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Chargements d'Algorithme de Bert pour l'Analyse des sentiments 

print("Loading sentiment analysis model...")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
#sentiment_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)


# les fonctions de transcription des audios vers les textes

def transcribe_audio(audio_path, sr=16000):
    try:
        audio, _ = librosa.load(audio_path, sr=sr)
        input_values = asr_processor(audio, sampling_rate=sr, return_tensors="pt").input_values
        with torch.no_grad():
            logits = asr_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = asr_processor.decode(predicted_ids[0])
        return transcription.lower()
    except Exception as e:
        print(f"Erreur de transcription : {e}")
        return 

# les fonctions de predictions sentiments

def predict_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label']
    if label == 'POSITIVE':
        return 'satisfait'
    elif label == 'NEGATIVE':
        return 'mécontent'
    else:
        return 'neutre'

# Endpoint principal
@app.post("/analyze/")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        # Sauvegarde temporaire du fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Transcription
        transcription = transcribe_audio(tmp_path)

        if transcription.startswith("Erreur"):
            return JSONResponse(content={"error": transcription}, status_code=400)

        # Analyse du sentiment
        sentiment = predict_sentiment(transcription)

        return {
            "transcription": transcription,
            "sentiment": sentiment
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Lancement local (exécuté uniquement si ce fichier est run directement)
if __name__ == "__main__":
    uvicorn.run("pipelines:app", host="127.0.0.1", port=8000)
