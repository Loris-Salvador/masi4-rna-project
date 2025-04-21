# Enoncé

[Enoncé](Enonce.pdf)

# Installation

**Attention problème avec `matplotlib` et une des dernières versions de python (`3.13`)**

**Version python conseillé = `3.8`**

https://www.python.org/downloads/release/python-3810/

## Cloner le projet 

```bash
git clone <url_distante>
```

## Création d'un environnement virtuel (optionnel)

```bash
python -m venv <nom_environnement>
```

Activation de l'environnement : 

```bash
env\Scripts\activate
```

## Installer les dépendances

```bash
pip install -r requirements.txt
```

# Contribution 

## Dépendances

Si vous ajoutez des dépendances ajoutez les aux `requirements.txt` pour faciliter le travail du prochain :)

Si vous travaillez dans un env virtuel (conseillé) cette commande prend toutes les dépendances et les écrit dans le fichier.

```bash
pip freeze > requirements.txt
```

Si pas d'env virtuel rajoutez à la main au fichier `requirements.txt` car sinon ca va écrire toutes les libs de votre pc dans le fichier (on veut seulement les nécessaires)


## Branches

Faites des branches quand vous travaillez pour que ça soit plus facile et propre 
