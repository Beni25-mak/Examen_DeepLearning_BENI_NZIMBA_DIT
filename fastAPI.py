from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline
import librosa
import torch
import tempfile
import os
import uvicorn
import tempfile


app = FastAPI()

# Chargement des modèles
print("Chargement du modèle Wav2Vec2...")
asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

print("Chargement du modèle d'analyse de sentiment...")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Fonction de transcription
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
        return ""

# Fonction de prédiction de sentiment
def predict_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label']
    if label == 'POSITIVE':
        return 'satisfait'
    elif label == 'NEGATIVE':
        return 'mécontent'
    else:
        return 'neutre'

# Endpoint API
@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    try:
        # Sauvegarder le fichier temporairement
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Transcrire l'audio
        transcription = transcribe_audio(tmp_path)
        if not transcription.strip():
            os.remove(tmp_path)
            return JSONResponse(content={"error": "Échec de la transcription"}, status_code=400)

        # Prédiction du sentiment
        sentiment = predict_sentiment(transcription)

        os.remove(tmp_path)
        return JSONResponse(content={
            "transcription": transcription,
            "sentiment": sentiment
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Lancement local (exécuté uniquement si ce fichier est run directement)
if __name__ == "__main__":
    uvicorn.run("pipelines:app", host="127.0.0.1", port=8000)