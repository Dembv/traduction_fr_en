import streamlit as st
import traduction as tr
import pandas as pd

st.set_page_config(
    page_title = "Tr'App",   
    page_icon = ":shark:",
)

taille = 5000
test_taille = 0.1

dataset = tr.charger_dataset("dataset/eng-french.csv", taille)

dataset = tr.melanger_dataset(dataset)

dataset = tr.nettoyer_dataset(dataset)
dataset = dataset.values
dataset = dataset[:taille]

train, test = tr.diviser_train_test(dataset, taille, test_taille)

train_source, train_cible, test_source, test_cible = tr.preparer_source_cible(dataset, train, test, "Fr")

idx_source, idx_cible = tr.choisir_source_cible("Fr") 

source_tokenizer = tr.creer_tokenizer(tr.get_lignes_colonnes(dataset, idx_source))
source_vocab_size = tr.get_vocab_size(source_tokenizer)

source_long = tr.longueur_max(tr.get_lignes_colonnes(dataset, idx_source))

cible_tokenizer = tr.creer_tokenizer(tr.get_lignes_colonnes(dataset, idx_cible))
cible_vocab_size = tr.get_vocab_size(cible_tokenizer)

cible_long = tr.longueur_max(tr.get_lignes_colonnes(dataset, idx_cible))

model = tr.creer_modele(source_vocab_size, cible_vocab_size, source_long, cible_long, 256)

model = tr.compiler_modele(model, 'adam', 'categorical_crossentropy')

history = tr.entrainer_modele(model, train_source, train_cible)

st.markdown('# :blue[Bienvenue] :red[dans] :red[Tr\'App]')
st.markdown("##### Votre application de traduction automatique")
st.divider()
#choix = st.radio("", ["Français -> Anglais", "English -> French"])
#if choix == "Français -> Anglais":
name = st. text_input("")
st.button("Traduire")

st.divider()
st.markdown("##### Traduction : ")
if name:
	#name = "Je suis sage"
	name = tr.nettoyer_phrase(name)
	#st.write(name)
	aname = tr.encoder_sequences(source_tokenizer, tr.longueur_max(name), name)
	name = tr.predire_sequence(model, cible_tokenizer, name)
	st.markdown(name)
#else:
 #   name = st. text_input("Text to translate : ")
  #  st.divider()
    #st.write("Result : ")


