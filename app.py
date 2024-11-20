from flask import Flask, request, render_template
from huggingface_hub import login
from googletrans import Translator
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import nltk

# Initialize the Flask app
app = Flask(__name__)

# Hugging Face login (you can also manage this through environment variables)
login(token="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")

# Initialize the models and tokenizers
translator = Translator()

# Load paraphrasing model
model_name = "Vamsi/T5_Paraphrase_Paws"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load grammar correction model
corrector_model = "prithivida/grammar_error_correcter_v1"
tokenizer_gc = AutoTokenizer.from_pretrained(corrector_model)
model_gc = AutoModelForSeq2SeqLM.from_pretrained(corrector_model)

# Initialize sentence transformer model
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Light model for embedding

# Download necessary NLTK data
nltk.download('punkt')

def translate_to_english(text):
    translation = translator.translate(text, src='hi', dest='en')
    return translation.text

def paraphrase(text):
    inputs = tokenizer.encode("paraphrase: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
    paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased_text

def grammar_correction(text):
    inputs = tokenizer_gc.encode(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model_gc.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
    corrected_text = tokenizer_gc.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def translate_to_hindi(text):
    translation = translator.translate(text, src='en', dest='hi')
    return translation.text

def process_hindi_paragraph(text):
    # Step 1: Translate Hindi to English
    english_text = translate_to_english(text)
    # Step 2: Paraphrase the English text
    paraphrased_text = paraphrase(english_text)
    # Step 3: Correct the Grammar of the Paraphrased text
    corrected_text = grammar_correction(paraphrased_text)
    # Step 4: Translate back to Hindi
    final_hindi_text = translate_to_hindi(corrected_text)
    return final_hindi_text

def calculate_ngram_similarity(text1, text2, n=1):
    # Tokenize text into words
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    
    # Create n-grams (unigrams or bigrams)
    ngrams1 = list(ngrams(tokens1, n))
    ngrams2 = list(ngrams(tokens2, n))
    
    # Calculate the number of common n-grams
    intersection = len(set(ngrams1).intersection(set(ngrams2)))
    union = len(set(ngrams1).union(set(ngrams2)))
    
    # Calculate similarity score as a ratio of intersection to union
    return intersection / union if union != 0 else 0

def validate_output(original_hindi_text, processed_hindi_text):
    # Translate both texts to English for comparison
    original_translation = translate_to_english(original_hindi_text)
    processed_translation = translate_to_english(processed_hindi_text)
    
    # Obtain embeddings for cosine similarity
    original_embedding = sentence_model.encode([original_translation])
    processed_embedding = sentence_model.encode([processed_translation])
    
    # Calculate cosine similarity
    cosine_similarity_score = cosine_similarity(original_embedding, processed_embedding)[0][0]
    
    # Calculate unigram and bigram similarity
    unigram_similarity = calculate_ngram_similarity(original_translation, processed_translation, n=1)
    bigram_similarity = calculate_ngram_similarity(original_translation, processed_translation, n=2)
    
    return {
        "original_translation": original_translation,
        "processed_translation": processed_translation,
        "cosine_similarity_score": cosine_similarity_score,
        "unigram_similarity": unigram_similarity,
        "bigram_similarity": bigram_similarity
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Get text from form
    hindi_text = request.form['hindi_text']
    
    # Process the Hindi paragraph
    processed_hindi_text = process_hindi_paragraph(hindi_text)
    
    # Validate output and get similarity metrics
    validation_results = validate_output(hindi_text, processed_hindi_text)
    
    # Calculate word counts
    original_word_count = len(hindi_text.split())
    processed_word_count = len(processed_hindi_text.split())
    
    # Render the result with similarity scores and word counts
    return render_template(
        'index.html', 
        input_text=hindi_text, 
        output_text=processed_hindi_text, 
        similarity_score=validation_results["cosine_similarity_score"], 
        unigram_similarity=validation_results["unigram_similarity"],
        bigram_similarity=validation_results["bigram_similarity"],
        original_word_count=original_word_count, 
        processed_word_count=processed_word_count
    )

if __name__ == '__main__':
    app.run(debug=True)