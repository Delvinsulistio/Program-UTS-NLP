"""
SentimentAI — Steam Review Analyzer  (Stacking Ensemble + Configurable Models)
"""
from flask import Flask, render_template, request, jsonify
import joblib, re, os, numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from collections import Counter

nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords',    quiet=True)
nltk.download('wordnet',      quiet=True)

app = Flask(__name__)

# ── Load models ──────────────────────────────────────────────────
MODEL_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
tfidf        = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))
svm_model    = joblib.load(os.path.join(MODEL_DIR, 'svm_model.joblib'))
rf_model     = joblib.load(os.path.join(MODEL_DIR, 'rf_model.joblib'))
bert_clf     = joblib.load(os.path.join(MODEL_DIR, 'bert_classifier.joblib'))
meta_learner = joblib.load(os.path.join(MODEL_DIR, 'meta_learner.joblib'))
le           = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))
bert_model   = SentenceTransformer('all-MiniLM-L6-v2')
sia          = SentimentIntensityAnalyzer()

stop_words   = set(stopwords.words('english'))
lemmatizer   = WordNetLemmatizer()
label_names  = list(le.classes_)   # ['Negative', 'Neutral', 'Positive']


# ── Helpers ──────────────────────────────────────────────────────
def softmax(scores):
    arr = np.array(scores, dtype=float)
    arr -= arr.max()
    e = np.exp(arr)
    return (e / e.sum()).tolist()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split()
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

def get_detailed_sentiment(label, confidence):
    if label == 'Positive':
        if confidence > 85: return ('Sangat Positif', 'Enthusiastic',    '🤩', '#00e676')
        if confidence > 65: return ('Positif',        'Enjoying It',     '😊', '#34d399')
        return                     ('Cenderung Positif','Leaning Positive','🙂', '#6ee7b7')
    elif label == 'Negative':
        if confidence > 85: return ('Sangat Negatif', 'Frustrated',      '😡', '#ff4d6d')
        if confidence > 65: return ('Negatif',         'Disappointed',   '😞', '#f87171')
        return                     ('Cenderung Negatif','Leaning Negative','😕','#fca5a5')
    else:
        return ('Netral', 'Mixed Feelings', '😐', '#ffc400')

def get_recommendation(label, _):
    return {'Positive': ('Recommended',     '👍'),
            'Negative': ('Not Recommended', '👎')}.get(label, ('Mixed', '🤔'))

def get_review_quality(n):
    if n < 5:   return 'Very Brief'
    if n < 15:  return 'Short Review'
    if n < 50:  return 'Moderate Review'
    if n < 150: return 'Detailed Review'
    return 'Comprehensive Review'


# ── Routes ───────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data        = request.get_json()
        review_text = data.get('review', '')
        selected    = set(data.get('models', ['svm', 'rf', 'bert']))
        fusion      = data.get('fusion', 'stacking')   # stacking | average | vote

        if not review_text.strip():
            return jsonify({'error': 'Review text is empty.'}), 400
        if not selected:
            return jsonify({'error': 'Enable at least one model in Settings.'}), 400

        # ─ Preprocessing ─
        cleaned         = preprocess_text(review_text)
        word_count      = len(review_text.split())
        clean_word_count = len(cleaned.split()) if cleaned.strip() else 0
        vader           = sia.polarity_scores(review_text)

        word_sentiments = [
            {'word': w, 'score': round(sia.polarity_scores(w)['compound'], 3)}
            for w in review_text.split()[:80]
        ]

        if not cleaned.strip():
            return jsonify({
                'result': {
                    'sentiment': 'Neutral', 'detailed_label': 'Netral',
                    'detailed_tone': 'Mixed Feelings', 'emoji': '😐',
                    'color': '#ffc400', 'confidence': 0,
                    'recommendation': 'Mixed', 'rec_emoji': '🤔',
                    'review_quality': get_review_quality(word_count),
                    'fusion_method': 'N/A', 'models_used': list(selected),
                    'class_probabilities': {'Negative': 0, 'Neutral': 100, 'Positive': 0}
                },
                'steps': {
                    'preprocessing': {'original': review_text, 'cleaned': '',
                                      'word_count': word_count, 'clean_word_count': 0},
                    'vader': vader, 'models': {}, 'stacking': {},
                    'word_sentiments': word_sentiments
                }
            })

        # ─ Feature extraction ─
        feat_tfidf = tfidf.transform([cleaned])
        feat_bert  = bert_model.encode([cleaned]) if 'bert' in selected else None

        # ─ Individual model inference ─
        svm_decision = rf_proba_arr = bert_proba_arr = None
        models_output, all_probas = {}, []

        if 'svm' in selected:
            svm_decision = svm_model.decision_function(feat_tfidf)
            svm_pred     = le.inverse_transform([svm_model.predict(feat_tfidf)[0]])[0]
            svm_scores   = svm_decision[0].tolist()
            svm_p        = softmax(svm_scores)
            models_output['Linear SVM'] = {
                'prediction': svm_pred,
                'scores': dict(zip(label_names, [round(s, 4) for s in svm_scores]))
            }
            all_probas.append(svm_p)

        if 'rf' in selected:
            rf_proba_arr = rf_model.predict_proba(feat_tfidf)
            rf_pred      = le.inverse_transform([rf_model.predict(feat_tfidf)[0]])[0]
            rf_scores    = rf_proba_arr[0].tolist()
            models_output['Random Forest'] = {
                'prediction': rf_pred,
                'scores': dict(zip(label_names, [round(s, 4) for s in rf_scores]))
            }
            all_probas.append(rf_scores)

        if 'bert' in selected:
            bert_proba_arr = bert_clf.predict_proba(feat_bert)
            bert_pred      = le.inverse_transform([bert_clf.predict(feat_bert)[0]])[0]
            bert_scores    = bert_proba_arr[0].tolist()
            models_output['BERT'] = {
                'prediction': bert_pred,
                'scores': dict(zip(label_names, [round(s, 4) for s in bert_scores]))
            }
            all_probas.append(bert_scores)

        # ─ Fusion ─
        can_stack   = (fusion == 'stacking' and {'svm','rf','bert'} <= selected)
        stacked_len = None

        if can_stack:
            stacked      = np.hstack([svm_decision, rf_proba_arr, bert_proba_arr])
            final_idx    = meta_learner.predict(stacked)[0]
            final_proba  = meta_learner.predict_proba(stacked)[0].tolist()
            final_label  = le.inverse_transform([final_idx])[0]
            fusion_name  = 'Stacking Ensemble'
            stacked_len  = int(stacked.shape[1])
        elif fusion == 'vote':
            votes        = [label_names[p.index(max(p))] for p in all_probas]
            final_label  = Counter(votes).most_common(1)[0][0]
            final_proba  = [sum(p[i] for p in all_probas) / len(all_probas)
                            for i in range(len(label_names))]
            fusion_name  = 'Majority Vote'
        else:
            final_proba  = [sum(p[i] for p in all_probas) / len(all_probas)
                            for i in range(len(label_names))]
            final_label  = label_names[final_proba.index(max(final_proba))]
            fusion_name  = 'Avg Probability'

        final_conf = round(float(max(final_proba)) * 100, 1)
        detail     = get_detailed_sentiment(final_label, final_conf)
        rec        = get_recommendation(final_label, final_conf)

        return jsonify({
            'result': {
                'sentiment':          final_label,
                'detailed_label':     detail[0],
                'detailed_tone':      detail[1],
                'emoji':              detail[2],
                'color':              detail[3],
                'confidence':         final_conf,
                'class_probabilities': dict(zip(label_names,
                                               [round(float(p) * 100, 1) for p in final_proba])),
                'recommendation':     rec[0],
                'rec_emoji':          rec[1],
                'review_quality':     get_review_quality(word_count),
                'fusion_method':      fusion_name,
                'models_used':        sorted(list(selected))
            },
            'steps': {
                'preprocessing': {
                    'original':        review_text,
                    'cleaned':         cleaned,
                    'word_count':      word_count,
                    'clean_word_count': clean_word_count
                },
                'vader':  {k: round(v, 4) for k, v in vader.items()},
                'models': models_output,
                'stacking': {
                    'fusion_method':      fusion_name,
                    'can_stack':          can_stack,
                    'meta_input_length':  stacked_len,
                    'final_probabilities': dict(zip(label_names,
                                                    [round(float(p), 4) for p in final_proba]))
                },
                'word_sentiments': word_sentiments
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n SentimentAI — Steam Review Analyzer")
    print("=" * 45)
    print(" http://localhost:5000")
    print("=" * 45 + "\n")
    app.run(debug=True, port=5000)
