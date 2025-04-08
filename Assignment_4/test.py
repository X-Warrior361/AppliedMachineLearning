import time
import pytest
import subprocess
import docker
from app import app


def test_docker():
    # Set image and container names
    image_name = "spam-classifier"
    container_name = "spam-classifier-test"
    
    client = None
    container = None
    
    try:  
    # Build Docker image
        build_cmd = f"docker build -t {image_name} ."
        subprocess.run(build_cmd, shell=True, check=True)

        # Run Docker container
        run_cmd = f"docker run -d -p 5000:5000 --name {container_name}  {image_name}"
        subprocess.run(run_cmd, shell=True, check=True) 

        # Give the container time to start
        time.sleep(10)

        # Connect to Docker API for clean termination later
        client = docker.from_env()
        container = client.containers.get(container_name)

        with app.test_client() as client:
        # Test endpoint with spam text
            spam_response = client.post(
                    '/score', 
                    json={'text': 'well done england get official poly ringtone colour flag yer mobile text tone flag optout txt eng stop box wwx £'},
                )

            # Validate response
            assert spam_response.status_code == 200
            response_data = spam_response.get_json()

            # Check response structure
            assert 'prediction' in response_data
            assert 'propensity' in response_data

            # Check prediction is True for spam
            assert response_data['prediction'] is True
            assert 0 <= response_data['propensity'] <= 1

            # Test non-spam text
            non_spam_response = client.post(
                    '/score', 
                    json={'text': 'Hi Mom, how are you today?'},
                )

            # Validate response
            assert non_spam_response.status_code == 200
            non_spam_data = non_spam_response.get_json()

            # Check prediction is False for non-spam
            assert non_spam_data['prediction'] is False
            assert 0 <= non_spam_data['propensity'] <= 1
    
    except Exception as e:
        pytest.fail(f"Docker test failed with error: {e}")
    
    finally:
        # Clean up - stop and remove container
        if container:
            container.stop()
            container.remove()
        elif client:
            # Try to clean up with command line if Docker API failed
            subprocess.run(
                f"docker stop {container_name} && docker rm {container_name}",
                shell=True,
                stderr=subprocess.DEVNULL
            )