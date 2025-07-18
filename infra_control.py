import subprocess
import shutil
import os
import time
from datetime import datetime

def log(msg):
    now = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p %z")
    print(f"[{now}] {msg}")

def clear_and_make_dirs():
    for d in ['aug_images', 'inference_results', 'coco_results']:
        path = os.path.join('results', d)
        if os.path.exists(path):
            log(f"Removing directory {path}")
            shutil.rmtree(path)
        log(f"Creating directory {path}")
        os.makedirs(path, exist_ok=True)

def get_container_id(service_name):
    """Get container ID for a given compose service"""
    try:
        # List containers related to compose project and service
        result = subprocess.run(
            ['docker', 'compose', 'ps', '-q', service_name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        container_id = result.stdout.strip()
        return container_id if container_id else None
    except subprocess.CalledProcessError as e:
        log(f"Error getting container ID: {e.stderr}")
        return None

def wait_for_container_healthy(container_id, timeout=120, interval=3):
    """Poll container health status until healthy, unhealthy, or timeout"""
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for container {container_id} to become healthy")

        try:
            inspect = subprocess.run(
                ['docker', 'inspect', '--format', '{{json .State.Health.Status}}', container_id],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
            )
            status = inspect.stdout.strip().strip('"')
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to inspect container health: {e.stderr}")

        log(f"Container {container_id} health status: {status}")

        if status == "healthy":
            log("Container is healthy.")
            return True
        elif status == "unhealthy":
            raise RuntimeError("Container healthcheck failed and is unhealthy.")

        time.sleep(interval)

def setup_infrastructure():
    clear_and_make_dirs()
    log("Starting services with Docker Compose...")
    cwd = os.getcwd()
    compose_dir = os.path.join(cwd, 'model_server')

    try:
        os.chdir(compose_dir)
        subprocess.run(['docker', 'compose', 'up', '-d', '--build'], check=True)
        service_name = "model-server"  # adjust to your service name in docker-compose.yml

        container_id = get_container_id(service_name)
        if not container_id:
            raise RuntimeError(f"Could not find container for service '{service_name}'")

        log(f"Waiting up to 2 minutes for container '{container_id}' to be healthy...")
        wait_for_container_healthy(container_id)
        log("Infrastructure setup completed successfully.")
    finally:
        os.chdir(cwd)

def teardown_infrastructure():
    log("Stopping services with Docker Compose...")
    cwd = os.getcwd()
    compose_dir = os.path.join(cwd, 'model_server')
    try:
        os.chdir(compose_dir)
        subprocess.run(['docker', 'compose', 'down'], check=True)
    finally:
        os.chdir(cwd)
    log("Infrastructure teardown completed.")
