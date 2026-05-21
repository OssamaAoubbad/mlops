FROM jenkins/jenkins:lts
USER root

# Install Docker CLI
RUN apt-get update && \
    apt-get install -y docker.io

# Drop back to the standard jenkins user (optional, but good practice. 
# However, for local Docker socket binding, root is often easier. We'll stay as root for local dev).