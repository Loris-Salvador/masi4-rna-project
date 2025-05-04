# Enoncé

[Enoncé](Enonce.pdf)

# Installation

**Attention problème avec `matplotlib` et une des dernières versions de python (`3.13`)**

**Version python conseillé = `3.8`**

https://www.python.org/downloads/release/python-3810/

## Cloner le projet 

```bash
git clone https://github.com/Loris-Salvador/masi4-rna-project.git
```

## Création d'un environnement virtuel (optionnel)

Dans le dossier du projet à la racine :

```bash
python -m venv env
```

Activation de l'environnement : 

```bash
env\Scripts\activate
```

## Installer les dépendances

Cette commande va lire les dépendances de `requirement.txt` et les installer

```bash
pip install -r requirements.txt
```

# Contributions

## Dépendances

Si vous ajoutez des dépendances ajoutez les aux `requirements.txt` pour faciliter le travail du prochain :)

Si vous travaillez dans un env virtuel (conseillé) cette commande prend toutes les dépendances et les écrit dans le fichier.

```bash
pip freeze > requirements.txt
```

Si pas d'env virtuel rajoutez à la main au fichier `requirements.txt` car sinon ca va écrire toutes les libs de votre pc dans le fichier (on veut seulement les nécessaires)


## Branches

Faites des branches quand vous travaillez pour que ça soit plus facile et propre 

## Pull Requests

Une fois le travail fini sur une branche (pas trop faire par branche) faire une pull request sur Github vers la main et ajouter des reviewers 


# Avancement  📋 

## Perceptron Simple

- Opérateur logique ET ✅ ***Loris***
- Classification de données linéairement séparables (table 2.9) ✅ ***Loris***
- Classfication de données non linéairement séparable (table 2.10) ✅ ***Loris***
- Régression linéaire (table 2.11) ✅ ***Loris***

## Perceptron Descente Gradient

- Opérateur logique ET ✅ ***Loris***
- Classification de données linéairement séparables (table 2.9) ✅ ***Loris***
- Classfication de données non linéairement séparable (table 2.10) ✅ ***Loris***
- Régression linéaire (table 2.11) ✅ ***Loris***


## Perceptron Adaline

- Opérateur logique ET ✅ ***Loris***
- Classification de données linéairement séparables (table 2.9) ✅ ***Loris***
- Classfication de données non linéairement séparable (table 2.10) ✅ ***Loris***
- Régression linéaire (table 2.11) ✅ ***Loris***

## Perceptron monocouche

- Classification à 3 classes (table 3.1) ✅ ***François***
- Classification à 4 classes (table 3.5) ✅ ***François***

## Perceptron multicouche

- Opérateur logique XOR (table 4.3) ✅ ***François***
- Classification à 2 classes non linéairement séparable (table 4.12) ✅ ***François***
- Classification à 3 classes non linéairement séparable (table 4.14) ✅ ***François***
- Régression non linéaire (table 4.17) ✅ ***François***

## DataSet réel

- Langue des signes ✅ ***Josué***
