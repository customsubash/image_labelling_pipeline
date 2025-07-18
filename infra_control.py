import subprocess
import shutil
import os
import time
from datetime import datetime
import requests


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


def wait_for_container_running(container_id, timeout=120, interval=3):
    """
    Wait until the container is in 'running' state.
    """
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError(
                f"Timeout: Container '{container_id}' did not start running in time.")

        try:
            result = subprocess.run(
                ['docker', 'inspect', '--format',
                    '{{.State.Status}}', container_id],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
            )
            status = result.stdout.strip()
            log(f"Container '{container_id}' status: {status}")

            if status == "running":
                log("Container is running.")
                return True

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Error inspecting container '{container_id}': {e.stderr.strip()}")

        time.sleep(interval)


def wait_for_http_ready(url, timeout=60, interval=3):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                print(f"[INFO] Model server is ready at {url}")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(interval)
    raise TimeoutError(f"Server at {url} not ready after {timeout} seconds")


def setup_infrastructure():
    clear_and_make_dirs()
    log("Starting services with Docker Compose...")
    cwd = os.getcwd()
    compose_dir = os.path.join(cwd, 'model_server')

    try:
        os.chdir(compose_dir)
        subprocess.run(['docker', 'compose', 'up',
                       '-d', '--build'], check=True)
        service_name = "model-server"  # adjust to your service name in docker-compose.yml

        container_id = get_container_id(service_name)
        if not container_id:
            raise RuntimeError(
                f"Could not find container for service '{service_name}'")

        log(
            f"Waiting up to 2 minutes for container '{container_id}' to be running...")
        # wait_for_container_healthy(container_id)
        # wait_for_container_running(container_id)
        HEALTH_CHECK_URL = 'http://localhost:8000/healthcheck'
        wait_for_http_ready(HEALTH_CHECK_URL)
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
