import os

# Dossier de base du projet, peu importe le fichier appelant
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chemins des fichiers (toujours relatifs à la racine projet)
TABLE_AND   = os.path.join(PROJECT_ROOT, "data", "table_AND.csv")
TABLE_2_9   = os.path.join(PROJECT_ROOT, "data", "table_2_9.csv")
TABLE_2_10  = os.path.join(PROJECT_ROOT, "data", "table_2_10.csv")
TABLE_2_11  = os.path.join(PROJECT_ROOT, "data", "table_2_11.csv")