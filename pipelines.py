import librosa
import torch
import gradio as gr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline

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

# Fonction finale pour Gradio

def process_audio_gradio(audio_file):
    transcription = transcribe_audio(audio_file)
    if transcription.strip() == "":
        return "Erreur lors de la transcription.", "", ""
    sentiment = predict_sentiment(transcription)
    return transcription, sentiment


# Interface Gradio
iface = gr.Interface(
    fn=process_audio_gradio,
    inputs=gr.Audio(type="filepath", label="Téléverser un fichier audio (.wav)"),
    outputs=[
        gr.Textbox(label="Texte transcrit"),
        gr.Textbox(label="Sentiment détecté")
    ],
    title="Détection automatique de sentiment à partir d'un appel vocal",
    description="Cette application transcrit un fichier audio (voix) en texte avec Wav2Vec2 et analyse le sentiment avec BERT."
)

iface.launch()
