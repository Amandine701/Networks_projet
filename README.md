## Comment faire fonctionner ce repo sur votre machine ?

Assurez-vous d'avoir `uv` d'installé:
```{bash}
pip install uv
```

Utilisez ces commandes dans l'ordre

```{bash}
git clone https://github.com/Amandine701/Networks_projet.git
cd Networks_projet/
uv sync
```

Pour exécuter les cellules du notebook, pensez à sélectionner l'interpréteur associé à l'environnement virtuel créé pour ce notebook. Voici la méthode si vous êtes sur VSCode et que l'environnement n'est pas détecté directement :
- Appuyez sur la touche `F1`
- Recherchez `Interpreter`
- Sélectionnez `Python: Select Interpreter > Enter interpreter path... > Find...`
- Sélectionnez le fichier situé à l'adresse `Networks_projet/.venv/bin/python`