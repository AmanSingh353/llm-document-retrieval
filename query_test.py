import requests

url = "http://localhost:5000/query"
payload = {
    # your payload here
}

try:
    response = requests.post(url, json=payload)
    response.raise_for_status()  # Raise HTTPError for bad responses

    print("Status Code:", response.status_code)
    print("Raw Response:", response.text)

    if response.headers.get("Content-Type", "").startswith("application/json"):
        print("JSON Response:", response.json())
    else:
        print("Response is not JSON.")
except requests.exceptions.ConnectionError:
    print(f"❌ Could not connect to server at {url}. Is the server running?")
except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
except Exception as e:
    print(f"An error occurred: {e}")
