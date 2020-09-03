import os
from flask import Flask, render_template, request, send_from_directory
import numpy as np
import pandas as pd
import spacy
import re

app = Flask(__name__)

STATIC_FOLDER = 'static'
# Path to the folder to store uploaded subtitle file
UPLOAD_FOLDER = STATIC_FOLDER + '/uploads'



def nlpparse(fullpath):
    #Load subtitles
    data = request.files['subtitle'].read()
    
    #Strip unwanted characters
    l = [re.sub(r'({.*})|(\|-)|(\|)|', '', i) for i in data]
    s = str(l)
    
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(s)
    tokens_pos = [token.pos_ for token in doc]
    tokens_text = [token.text for token in doc]
    tokens_lemma = [token.lemma_ for token in doc]
    df = zip(tokens_pos, tokens_text, tokens_lemma)
    df = pd.DataFrame(df)
    df.columns = ('partofspeech', 'text', 'lemma')
    #drop PUNCT and SYM
    df = df[df.partofspeech != 'PUNCT']
    df = df[df.partofspeech != 'SYM']
    pos_count = df.groupby('partofspeech')['text'].nunique()
    vocab_count = pos_count.sum()
    
    return pos_count, vocab_count

#Landing page
@app.route('/')
def index():
    return render_template('index.html')

# Process file and run nlpparse
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['subtitle']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)
        
    #return redirect(url_for('index'))

        pos_count, vocab_count = nlpparse(fullname)
        pos_count = pd.DataFrame(pos_count)
        return render_template('predict.html', vocab_count = vocab_count, pos_count = pos_count, tables=[pos_count.to_html(classes='data')], titles=pos_count.columns.values)

@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def create_app():
    load__model()
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0')
