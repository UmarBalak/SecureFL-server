import requests
import os
import uuid
from dotenv import load_dotenv

# Register client with server
def register_client(csn, ADMIN_API_KEY, API_URL):
    payload = {
        "csn": csn,
        "admin_api_key": ADMIN_API_KEY
    }
    try:
        # response = requests.post(f"http://localhost:{8000}/register", json=payload)
        response = requests.post(f"{API_URL}", json=payload)
        if response.status_code == 200:
            data = response.json().get("data", {})
            client_id = data.get("client_id")
            api_key = data.get("api_key")
            if client_id and api_key:
                print(f"Client registered successfully: CSN={csn}, Client ID={client_id}, api key={api_key}")
            else:
                print("Invalid response from server")
        else:
            print(f"Failed to register client: {response.json().get('detail', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to the server: {e}")

# Generate a unique CSN
def generate_csn():
    return f"CSN-{uuid.uuid4()}"

if __name__ == "__main__":
    # Load environment variables
    load_dotenv(dotenv_path='.env.server')
    ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
    API_URL = os.getenv("CLIENT_REGISTRATION_ENDPOINT")
    if not ADMIN_API_KEY or not API_URL:
        print("ADMIN_API_KEY or CLIENT_REGISTRATION_ENDPOINT not found in .env.server file")
        exit(1)
    # Option to generate CSN automatically for testing
    auto_generate = input("Do you want to auto-generate a CSN? (y/n): ").strip().lower()
    if auto_generate == "y":
        csn = generate_csn()
        print(f"Generated CSN: {csn}")
    else:
        csn = input("Enter the CSN (Client Serial Number): ").strip()
    
    register_client(csn, ADMIN_API_KEY, API_URL)
