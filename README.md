# Enoncé

[Enoncé](Enonce.pdf)

# Installation

**Attention problème avec `matplotlib` et la dernière version de python**

**Version python conseillé = `3.8`**

## Cloner le projet 

```bash
git clone <url_distante>
```

## Création d'un environnment virtuel (optionnel)

```bash
python -m venv <nom_environnment>
```

Activation de l'environnement : 

```bash
env\Scripts\activate
```

## Installer les dépendances

```bash
pip install -r requirements.txt
```

# Update 

Si vous ajoutez des dépendances ajoutez les aux `requirements.txt` pour faciliter le travail du prochain :)

```bash
pip freeze > requirements.txt
```