# client.py
import requests
import json
import argparse
import time
import os
# replace this URL with your exposed URL from the API builder. The URL looks like this
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:8000")

def main():
    parser = argparse.ArgumentParser(description="Send a query to the LitServe server.")
    parser.add_argument("--query", type=str, required=True, help="The query text to send to the server.")
    
    args = parser.parse_args()
    
    payload = {
        "query": args.query
    }
    
    try:
        response = requests.post(f"{SERVER_URL}/predict", json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        data = response.json()
        output = data.get("output")
        if isinstance(output, dict) and "raw" in output:
            result = output["raw"]
        else:
            result = output
        if result is None:
            result = ""

        for token in str(result).split():
            print(token, end=" ", flush=True)
            time.sleep(0.05)

        citations = data.get("citations", [])
        if citations:
            print("\n\nSources:")
            for item in citations:
                source = item.get("source", "unknown")
                page_start = item.get("page_start")
                page_end = item.get("page_end")
                if page_start and page_end and page_start != page_end:
                    page_label = f"pages {page_start}-{page_end}"
                elif page_start:
                    page_label = f"page {page_start}"
                else:
                    page_label = "page ?"
                print(f"- {source} ({page_label})")

        # print(json.dumps(result, indent=2))
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")

if __name__ == "__main__":
    main()
