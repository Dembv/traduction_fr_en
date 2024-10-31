import warnings
warnings.filterwarnings('ignore')
import string
import re
from unicodedata import normalize
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential,load_model
from keras.layers import LSTM,Dense,Embedding,RepeatVector,TimeDistributed
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
from string import punctuation
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

total_sentences = 500

def upload_dataset(v_total_sentences):
    # Charger le jeu de données
    dataset = pd.read_csv("dataset/eng-french.csv", nrows = v_total_sentences)
    return dataset

dataset = upload_dataset(total_sentences)

# Quelle proportion de phrases sera utilisée pour l'ensemble de tests
test_proportion = 0.2
train_test_threshold = int( (1-test_proportion) * total_sentences)

# Mélanger le jeu de données
dataset = dataset.sample(frac=1, random_state=0)
dataset.iloc[1000:1010]

def clean(string):
    # Nettoyer la phrase
    string = string.replace("\u202f"," ") # Replace no-break space with space
    string = string.lower()

    # Supprimez la ponctuation et les chiffres
    for p in punctuation + "«»" + "0123456789":
        string = string.replace(p," ")

    string = re.sub('\s+',' ', string)
    string = string.strip()

    return string

# Nettoyer les phrases
dataset["English words/sentences"] = dataset["English words/sentences"].apply(lambda x: clean(x))
dataset["French words/sentences"] = dataset["French words/sentences"].apply(lambda x: clean(x))

# Sélectionnez une partie du jeu de données
dataset = dataset.values
dataset = dataset[:total_sentences]

# diviser en train/test
train, test = dataset[:train_test_threshold], dataset[train_test_threshold:]

# Définir le nom de la source et de la cible
source_str, target_str = "French", "English"

# L'index dans le tableau numpy de la source et de la cible
idx_src, idx_tar = 1, 0

# Afficher le résultat après nettoyage
pd.DataFrame(dataset[1000:1010])

def create_tokenizer(lines):
    # initialiser un tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_len(lines):
    # longueur maximale de la phrase
    return max(len(line.split()) for line in lines)

def encode_sequences(tokenizer, length, lines):
    # encode and pad sequences
    X = tokenizer.texts_to_sequences(lines) # integer encode sequences
    X = pad_sequences(X, maxlen=length, padding='post') # pad sequences with 0 values
    return X

def encode_output(sequences, vocab_size):
    # one hot encode target sequence
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = np.array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y

# Préparer le tokenizer cible
tar_tokenizer = create_tokenizer(dataset[:, idx_tar])
tar_vocab_size = len(tar_tokenizer.word_index) + 1
tar_length = max_len(dataset[:, idx_tar])

# Préparer le tokenizer source
src_tokenizer = create_tokenizer(dataset[:, idx_src])
src_vocab_size = len(src_tokenizer.word_index) + 1
src_length = max_len(dataset[:, idx_src])

# Préparer les données dd'entraînement
trainX = encode_sequences(src_tokenizer, src_length, train[:, idx_src])
trainY = encode_sequences(tar_tokenizer, tar_length, train[:, idx_tar])
trainY = encode_output(trainY, tar_vocab_size)

# Préparer les données de test
testX = encode_sequences(src_tokenizer, src_length, test[:, idx_src])
testY = encode_sequences(tar_tokenizer, tar_length, test[:, idx_tar])
testY = encode_output(testY, tar_vocab_size)

print(trainY)
# Créer et entraîner le modèle

def create_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    # Créer le modèle
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model

# Créer un modèle
model = create_model(src_vocab_size, tar_vocab_size, src_length, tar_length, 256)


model.compile(optimizer='adam', loss='categorical_crossentropy')

history = model.fit(trainX,
          trainY,
          epochs=200,
          batch_size=64,
          validation_split=0.1,
          verbose=1,
          callbacks=[
                        EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
            ])

# Résultat sur les données de test

pd.DataFrame(history.history).plot()
plt.title("Loss")
plt.show()

def word_for_id(integer, tokenizer):
    # Faire correspondre un entier à un mot
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_seq(model, tokenizer, source):
    # générer une cible à partir d'une séquence source
    prediction = model.predict(source, verbose=0)[0]
    integers = [np.argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)

def compare_prediction(model, tokenizer, sources, raw_dataset, limit=20):
    # évaluer un modèle
    actual, predicted = [], []
    src = f'{source_str.upper()} (SOURCE)'
    tgt = f'{target_str.upper()} (TARGET)'
    pred = f'AUTOMATIC TRANSLATION IN {target_str.upper()}'
    print(f'{src:30} {tgt:25} {pred}\n')

    for i, source in enumerate(sources): # traduire le texte source codé
        source = source.reshape((1, source.shape[0]))
        translation = predict_seq(model, tar_tokenizer, source)
        raw_target, raw_src = raw_dataset[i]
        print(f'{raw_src:30} {raw_target:25} {translation}')
        if i >= limit: # Afficher une partie du résultat
            break

# tests sur quelques données d'entraînement
print('### Result on the Training Set ###')
compare_prediction(model, tar_tokenizer, trainX, train)

# tests sur quelques données de tests
print('\n\n### Result on the Test Set ###')
compare_prediction(model, tar_tokenizer, testX, test)

# Évaluation des prédictions avec BLEU 
## Algorithme
# Le calcul du score BLEU prend du temps

def bleu_score(model, tokenizer, sources, raw_dataset):
    # Obtenir le score bleu d'un modèle
    actual, predicted = [], []
    for i, source in enumerate(sources):
        # traduire le texte source codé
        source = source.reshape((1, source.shape[0]))
        translation = predict_seq(model, tar_tokenizer, source)
        raw_target, raw_src = raw_dataset[i]
        actual.append([raw_target.split()])
        predicted.append(translation.split())

    bleu_dic = {}
    bleu_dic['1-grams'] = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    bleu_dic['1-2-grams'] = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    bleu_dic['1-3-grams'] = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
    bleu_dic['1-4-grams'] = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

    return bleu_dic

# Calculer le score BLEU
bleu_train = bleu_score(model, tar_tokenizer, trainX, train)
bleu_test = bleu_score(model, tar_tokenizer, testX, test)

plt.bar(x = bleu_train.keys(), height = bleu_train.values())
plt.title("BLEU Score with the training set")
plt.ylim((0,1))
plt.show()

plt.bar(x = bleu_test.keys(), height = bleu_test.values())
plt.title("BLEU Score with the test set")
plt.ylim((0,1))
plt.show()

