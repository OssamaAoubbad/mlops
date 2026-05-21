# CI/CD Pipeline - Corrections effectuées

## Problèmes identifiés et résolus

### 1. **Problème principal : `run_id = null` lors de l'exécution sur Jenkins**

**Cause:** 
Le Jenkinsfile tentait d'extraire le `run_id` en utilisant une requête MLflow complexe dans un one-liner shell, ce qui était peu fiable :
```groovy
sh "${env.VENV_PY} -c \"from madewithml.config import MLFLOW_TRACKING_URI, mlflow; mlflow.set_tracking_uri(MLFLOW_TRACKING_URI); runs = mlflow.search_runs(experiment_names=['mlops-project'], order_by=['metrics.val_loss ASC']); print(runs.iloc[0].run_id)\" > run_id.txt"
```

**Problèmes:**
- Import MLflow incorrect dans la commande Python inline
- Pas de gestion d'erreurs
- Dépend de pandas pour accéder à `.iloc[0]`

**Solution implémentée ✓**
Extraction du `run_id` directement depuis `results.json` (qui est déjà produit par le script train.py) :
```groovy
script {
    try {
        def resultsFile = readFile('results.json')
        def results = new groovy.json.JsonSlurper().parseText(resultsFile)
        env.RUN_ID = results.run_id
        echo "✓ Successfully captured run_id: ${env.RUN_ID}"
    } catch (Exception e) {
        error("Failed to parse results.json or extract run_id: ${e.message}")
    }
}
```

### 2. **Vérification du run_id pour éviter null**

Ajout de validation stricte après l'extraction :
```groovy
if [ -z "${RUN_ID}" ] || [ "${RUN_ID}" = "null" ]; then
    echo "ERROR: run_id is empty or null!"
    cat results.json
    exit 1
fi
```

### 3. **Amélioration du Serve stage**

- Corrigé `kill $SERVER_PID || true` pour éviter une erreur si le processus est déjà terminé
- Ajout de la syntaxe correcte `${RUN_ID}` dans le context shell

### 4. **Débuggage amélioré**

Ajout de logs de diagnostic :
```groovy
sh "echo '=== Results.json content ===' && cat results.json"
sh "echo '=== Evaluation results ===' && cat eval-results.json"
```

## Structure du pipeline correctif

**Stage Train:**
- ✓ Entraîne le modèle
- ✓ Produit `results.json` contenant le `run_id`
- ✓ Affiche le contenu de `results.json` pour vérification
- ✓ Parse le JSON et extrait `run_id` avec gestion d'erreurs
- ✓ Valide que `run_id` n'est pas vide ou null

**Stage Evaluate:**
- ✓ Utilise `${env.RUN_ID}` correctement
- ✓ Affiche les résultats d'évaluation

**Stage Serve:**
- ✓ Reçoit le `run_id` correct du stage Train
- ✓ Gestion propre de l'arrêt du serveur

## Prochaines étapes de vérification

1. **Vérifiez que `train.py` produit correctement le `run_id`**
   - Le script train.py line ~240 : `"run_id": run.info.run_id` ✓

2. **Testez le pipeline Jenkins**
   - Déclenchez une nouvelle exécution du job
   - Vérifiez les logs du stage "Train" pour voir le contenu de results.json
   - Confirmez que `run_id` est extracté correctement

3. **Monitoring MLflow**
   - Les runs devraient apparaître dans MLflow tracking
   - Vérifiez que le `MLFLOW_TRACKING_URI` pointe au bon répertoire

## Fichiers modifiés

- [Jenkinsfile](Jenkinsfile) - Corrections du pipeline CI/CD
