from sentence_transformers import SentenceTransformer
import numpy as np
from langdetect import detect
from flask import Flask, request
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

# Load business data
with open("business_data.txt", "r") as file:
    sentences = [line.strip() for line in file if line.strip()]

# Load Hugging Face model
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
embeddings = model.encode(sentences)

# Response function
def get_response(query):
    lang = detect(query)
    query_embedding = model.encode([query])
    similarities = np.dot(query_embedding, embeddings.T)
    best_match_idx = np.argmax(similarities)
    response = sentences[best_match_idx]
    if lang == "si":
        return f"[Sinhala placeholder]: {response}"
    return response

# Twilio credentials (replace with your own)
account_sid = "##############################"
auth_token = "#############################"
client = Client(account_sid, auth_token)

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    incoming_msg = request.values.get("Body", "").strip()
    response_text = get_response(incoming_msg)
    msg = MessagingResponse()
    msg.message(response_text)
    return str(msg)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
