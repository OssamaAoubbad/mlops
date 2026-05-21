pipeline {

    agent {
        docker {
            image 'python:3.10.0-slim'
            reuseNode true
        }
    }

    options {
        timestamps()
    }

    environment {
        VENV_DIR = ".venv"
        VENV_PY  = ".venv/bin/python"
        RUN_ID   = ""
    }

    stages {

        stage("Checkout") {
            steps {
                checkout scm
            }
        }

        stage("Setup") {
            steps {
                sh '''
                    apt-get update
                    apt-get install -y gcc

                    python -m venv ${VENV_DIR}

                    ${VENV_PY} -m pip install --upgrade pip
                    ${VENV_PY} -m pip install -r requirements.txt
                '''
            }
        }

        stage("Tests") {
            steps {
                sh '''
                    if [ -d "tests" ] || [ -f "pytest.ini" ] || [ -f "pyproject.toml" ]; then
                        ${VENV_PY} -m pytest -q --junitxml=test-results.xml
                    else
                        echo "No tests found, skipping."
                    fi
                '''
            }
        }

        stage("Train") {
            steps {

                sh """
                    ${VENV_PY} -m madewithml.train \
                    --experiment-name mlops-project \
                    --num-epochs 1 \
                    --results-fp results.json
                """

                sh "echo '=== Results.json content ==='"
                sh "cat results.json"

                script {

                    if (!fileExists('results.json')) {
                        error("results.json not found!")
                    }

                    try {
                        def resultsFile = readFile('results.json')
                        def results = new groovy.json.JsonSlurper().parseText(resultsFile)

                        env.RUN_ID = results.run_id

                        echo "✓ Successfully captured run_id: ${env.RUN_ID}"

                    } catch (Exception e) {
                        error("Failed to parse results.json: ${e.message}")
                    }
                }

                sh '''
                    if [ -z "${RUN_ID}" ] || [ "${RUN_ID}" = "null" ]; then
                        echo "ERROR: run_id is empty!"
                        cat results.json
                        exit 1
                    fi

                    echo "Run ID validated: ${RUN_ID}"
                '''
            }
        }

        stage("Evaluate") {
            steps {

                sh """
                    ${VENV_PY} -m madewithml.evaluate \
                    --run-id ${RUN_ID} \
                    --dataset-loc datasets/holdout.csv \
                    --results-fp eval-results.json
                """

                sh "echo '=== Evaluation results ==='"
                sh "cat eval-results.json"
            }
        }

        stage("Serve") {
            steps {

                sh '''
                    ${VENV_PY} -m madewithml.serve \
                    --run_id ${RUN_ID} \
                    --host 127.0.0.1 \
                    --port 8000 &

                    SERVER_PID=$!

                    sleep 10

                    ${VENV_PY} -c "
import urllib.request
response = urllib.request.urlopen('http://127.0.0.1:8000/')
print(response.read().decode('utf-8'))
"

                    kill $SERVER_PID || true
                '''
            }
        }
    }

    post {

        always {

            archiveArtifacts artifacts: "**/results.json", allowEmptyArchive: true

            archiveArtifacts artifacts: "**/eval-results.json", allowEmptyArchive: true

            junit allowEmptyResults: true,
                   testResults: "**/test-results.xml"
        }
    }
}