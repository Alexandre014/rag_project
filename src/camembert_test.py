from sentence_transformers import SentenceTransformer, util

# Charger le modèle
model = SentenceTransformer("dangvantuan/sentence-camembert-base")

# Définir les phrases à comparer
phrase1 = """Pour échouer à obtenir le diplôme, il faut ne pas remplir les conditions suivantes : 
- Ne pas obtenir 120 crédits européens pour le DUT ou ne pas valider toutes les UE des six semestres (ou ne pas avoir droit à une dispense pour certaines UE) et ne pas présenter au moins une certification en langue anglaise pour le B.U.T. 
- Excéder le nombre maximum de redoublements autorisés (4 semestres sur l'ensemble du cursus). 
- Ne pas respecter les obligations d'assiduité, c'est-à-dire avoir trop d'absences injustifiées au-delà du seuil fixé par la spécialité. 
- Être refusé pour redoublement et ne pas pouvoir être réorienté après audition par le jury."""
phrase2 = "Pour échouer l'obtention du diplôme il faudrait ne pas avoir assez de crédit, ne pas respecter l'assiduité ou redoubler trop de fois"

# Obtenir les embeddings des phrases
embedding1 = model.encode(phrase1, convert_to_tensor=True)
embedding2 = model.encode(phrase2, convert_to_tensor=True)

# Calculer la similarité cosinus
similarity_score = util.pytorch_cos_sim(embedding1, embedding2)

print(f"Score de similarité : {similarity_score.item():.4f}")
