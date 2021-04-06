from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

t=pd.read_csv('data/train1.csv')

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(t['search_term'])

indices = pd.Series(t.index, index=t['search_term'])
all_titles = sorted(list(set(t['search_term'])))

warnings.filterwarnings("ignore")

def get_recommendations(title):
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x, reverse=True)
    sim_scores = sim_scores[1:11]
    product_indices = [i[0] for i in sim_scores]
    ttl = t['product_title'].iloc[product_indices]
    return_df = pd.DataFrame(columns=['Title'])
    return_df['Title'] = ttl
    return return_df

app = Flask(__name__)
Bootstrap(app)
@app.route('/')
def index():
    return render_template('index.html',products = all_titles)

@app.route('/predict', methods=['POST'])
def predict():

    m_name = request.form['choice']
    user_name = request.form['username']

    if m_name not in all_titles:
        return render_template('result.html',prediction = ['No Recommendation exist currently',''],username = user_name)
    else:
        result_final = get_recommendations(m_name)
        names = []
        for i in range(len(result_final)):
            names.append(result_final.iloc[i][0])

    return render_template('result.html',prediction = names,username = user_name)

if __name__ == '__main__':
	app.run(debug=True)