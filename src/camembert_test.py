from sentence_transformers import SentenceTransformer, util

# Charger le modèle
model = SentenceTransformer("dangvantuan/sentence-camembert-base")

# Définir les phrases à comparer
phrase1 = "Le chat dort sur le canapé."
phrase2 = "Un félin se repose sur le sofa."

# Obtenir les embeddings des phrases
embedding1 = model.encode(phrase1, convert_to_tensor=True)
embedding2 = model.encode(phrase2, convert_to_tensor=True)

# Calculer la similarité cosinus
similarity_score = util.pytorch_cos_sim(embedding1, embedding2)

print(f"Score de similarité : {similarity_score.item():.4f}")
