# schwabttk/schwab_auth.py
import os
import schwab
from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv()

APP_KEY = os.getenv("SCHWAB_APP_KEY")
APP_SECRET = os.getenv("SCHWAB_APP_SECRET")
CALLBACK_URL = os.getenv("SCHWAB_CALLBACK_URL")
TOKEN_PATH = os.getenv("SCHWAB_TOKEN_PATH")

def get_client():
    """
    Returns an authenticated Schwab client.
    First run : opens browser for OAuth login.
    After that: uses saved token.json automatically.
    """
    client = schwab.auth.easy_client(
        api_key=APP_KEY,
        app_secret=APP_SECRET,
        callback_url=CALLBACK_URL,
        token_path=TOKEN_PATH
    )
    return client

if __name__ == "__main__":
    client = get_client()
    print("Authentication successful!")

    # Quick test — get account numbers
    response = client.get_account_numbers()
    print(f"Status code: {response.status_code}")

    if response.status_code == 200:
        accounts = response.json()
        print(f"Linked accounts found: {len(accounts)}")
        for acct in accounts:
            print(f"  Account: {acct['accountNumber']}")
    else:
        print("Could not retrieve accounts.")