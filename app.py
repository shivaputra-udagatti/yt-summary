
from flask import Flask
from flask import render_template
import json
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import openai
import pinecone
from flask import request
import textwrap
from os import environ

app = Flask(__name__)



# retriever = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base')
# embed_dim = retriever.get_sentence_embedding_dimension()


# connect to new index
# index = pinecone.Index("youtube-search")

@app.route('/')
def home():
	return render_template('home.html',data='')

@app.route('/search', methods=['GET','POST'])
def search():
	if request.method == 'POST':
		query = request.form['query']
		xq = retriever.encode(query).tolist()
		xc = index.query(xq, top_k=5,include_metadata=True)
		cont = []
		summary_chunks=[]
		count =0
		for context in xc['matches']:
			summary = generate_summary_(context['metadata']['text'])
			summary_chunks.append(summary)
			cont.append(context['metadata'])		
		return render_template('home.html',data=(cont),summary=' '.join(summary_chunks))
	else:
		return render_template('home.html')

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

@app.route('/generate-summary',methods=['GET','POST'])
def video_summary():
	if request.method == 'POST':
		video_id= request.form['vid']
		transcript = get_video_transcript(video_id) #uemEfCCGw4E
		summary = generate_summary(transcript)
		return render_template('home.html',summary=summary,transcript=transcript)
	return render_template('home.html',summary='',transcript='')

@app.route('/video-transcript/<video_id>',methods=['GET','POST'])
def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except TranscriptsDisabled:
        return None

    text = " ".join([line["text"] for line in transcript])
    return text

def generate_summary_(text):
    openai.api_key = environ.get('API_KEY')
    instructions = "Please summarize the provided text"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": text}
        ],
        temperature=0.2,
        n=1,
        max_tokens=1000,
        presence_penalty=0,
        frequency_penalty=0.1,
    )
    return response.choices[0].message.content.strip()

def generate_summary(text):
	input_chunks = text_wrap(text)
	output_chunks = []
	openai.api_key = environ.get('API_KEY')
	for chunk in input_chunks:
		response = openai.Completion.create(
			engine="davinci",
			prompt=(f"Please summarize the following text:\n{chunk}\n\nSummary:"),
			temperature=0.5,
			max_tokens=250,n = 1,
			stop=None
			)
		summary = response.choices[0].text.strip()
		output_chunks.append(summary)
	return " ".join(output_chunks)

@app.route('/test')
def test():
	response = openai.Completion.create(
			engine="davinci",
			prompt=(f"Please more elaborate the summary "),
			temperature=0.5,
			max_tokens=400,n = 1,
			stop=None
			)
	return response.choices[0].text.strip()

def text_wrap(text):
	wrapper = textwrap.TextWrapper(width=5000)
	word_list = wrapper.wrap(text=text)
	return word_list

def split_text(text):
    max_chunk_size = 1000
    chunks = []
    current_chunk = ""
    for sentence in text.split("."):
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + "."
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "."
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# main driver function
if __name__ == '__main__':
	app.run(host='0.0.0.0',debug=True,port=8000)


