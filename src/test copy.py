import re
def clean_original_response(text):
    """Keep the first response before any rephrasing"""
    patterns = [
        r"(?i)^\**\s*réponse(\s+en\s+français)?(\s+(concise|finale|formatée))?\s*[:：]",
    ]

    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if any(re.match(pattern, line.strip()) for pattern in patterns):
            break
        cleaned.append(line)
    return "\n".join(cleaned).strip()

text = """
Les dates du stage doivent être fixées en fonction des contrats d'alternance. Il est impératif de signer un contrat avant le 15 octobre, et le stage doit se terminer au plus tard le 31 août.

**Réponse finale :**
Le stage doit commencer avant ou pendant la rentrée de septembre et se terminer au plus tard le 31 août.
"""
print(clean_original_response(text))