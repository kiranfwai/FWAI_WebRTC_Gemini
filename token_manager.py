"""
Meta/WhatsApp Token Manager
Auto-generates and refreshes access tokens using App ID and App Secret

Supports:
1. App Access Token generation (client_credentials)
2. Long-lived token exchange
3. Token caching and auto-refresh
"""

import os
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Token storage file
TOKEN_FILE = Path(__file__).parent / ".token_cache.json"

# Meta App credentials
META_APP_ID = os.getenv("META_APP_ID", "800389698460627")
META_APP_SECRET = os.getenv("META_APP_SECRET", "723307e7a6333fa54f72e07ac010e17c")

# WhatsApp Business Account ID (needed for some token operations)
WABA_ID = os.getenv("WABA_ID", "")


def log(message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [TOKEN] {message}")


class TokenManager:
    """Manages Meta/WhatsApp API access tokens"""

    def __init__(self):
        self.access_token = os.getenv("META_ACCESS_TOKEN", "")
        self.token_expiry = None
        self._load_cached_token()

    def _load_cached_token(self):
        """Load token from cache file if exists"""
        try:
            if TOKEN_FILE.exists():
                with open(TOKEN_FILE, 'r') as f:
                    data = json.load(f)
                    cached_token = data.get("access_token")
                    expiry_str = data.get("expiry")

                    if cached_token and expiry_str:
                        expiry = datetime.fromisoformat(expiry_str)
                        # Use cached token if not expired (with 1 hour buffer)
                        if expiry > datetime.now() + timedelta(hours=1):
                            self.access_token = cached_token
                            self.token_expiry = expiry
                            log(f"Loaded cached token (expires: {expiry.strftime('%Y-%m-%d %H:%M')})")
                            return
        except Exception as e:
            log(f"Error loading cached token: {e}")

    def _save_token_cache(self, token: str, expires_in: int = 5184000):
        """Save token to cache file"""
        try:
            expiry = datetime.now() + timedelta(seconds=expires_in)
            with open(TOKEN_FILE, 'w') as f:
                json.dump({
                    "access_token": token,
                    "expiry": expiry.isoformat(),
                    "created": datetime.now().isoformat()
                }, f, indent=2)
            self.access_token = token
            self.token_expiry = expiry
            log(f"Saved token to cache (expires: {expiry.strftime('%Y-%m-%d %H:%M')})")
        except Exception as e:
            log(f"Error saving token cache: {e}")

    async def get_valid_token(self) -> str:
        """Get a valid access token, refreshing if necessary"""
        # Check if current token is valid
        if self.access_token and self._is_token_valid():
            return self.access_token

        # Try to exchange for long-lived token
        log("Token expired or invalid, attempting to refresh...")
        new_token = await self._exchange_for_long_lived_token()

        if new_token:
            return new_token

        # If exchange fails, return current token (will fail with 401)
        log("Token refresh failed, using existing token")
        return self.access_token

    def _is_token_valid(self) -> bool:
        """Check if current token is likely valid"""
        if not self.access_token:
            return False
        if self.token_expiry and datetime.now() > self.token_expiry - timedelta(hours=1):
            return False
        return True

    async def _generate_app_access_token(self) -> str | None:
        """
        Generate App Access Token using client_credentials grant.
        This creates a token in the format: {app_id}|{app_secret}
        """
        if not META_APP_ID or not META_APP_SECRET:
            log("App ID or App Secret not configured")
            return None

        try:
            # Method 1: Direct client_credentials grant
            url = "https://graph.facebook.com/oauth/access_token"
            params = {
                "client_id": META_APP_ID,
                "client_secret": META_APP_SECRET,
                "grant_type": "client_credentials"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        new_token = data.get("access_token")

                        if new_token:
                            # App tokens don't expire, but we'll cache for 30 days
                            self._save_token_cache(new_token, 2592000)
                            log(f"Generated new App Access Token successfully")
                            return new_token
                    else:
                        error = await response.text()
                        log(f"App token generation failed ({response.status}): {error[:200]}")
        except Exception as e:
            log(f"App token generation error: {e}")

        # Method 2: Fallback - construct token directly
        # App Access Tokens can be constructed as: {app_id}|{app_secret}
        log("Trying direct app token construction...")
        direct_token = f"{META_APP_ID}|{META_APP_SECRET}"
        self._save_token_cache(direct_token, 2592000)
        return direct_token

    async def _generate_system_user_token(self) -> str | None:
        """
        Generate System User Access Token.
        Requires existing app token and system user ID.
        """
        if not META_APP_ID or not META_APP_SECRET:
            return None

        try:
            # First get app token
            app_token = f"{META_APP_ID}|{META_APP_SECRET}"

            # Get system users for this business
            url = f"https://graph.facebook.com/v21.0/{META_APP_ID}/system_users"

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params={"access_token": app_token},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        system_users = data.get("data", [])

                        if system_users:
                            system_user_id = system_users[0].get("id")
                            log(f"Found system user: {system_user_id}")

                            # Generate token for system user
                            token_url = f"https://graph.facebook.com/v21.0/{system_user_id}/access_tokens"
                            token_params = {
                                "business_app": META_APP_ID,
                                "scope": "whatsapp_business_management,whatsapp_business_messaging",
                                "access_token": app_token
                            }

                            async with session.post(
                                token_url,
                                params=token_params,
                                timeout=aiohttp.ClientTimeout(total=10)
                            ) as token_response:
                                if token_response.status == 200:
                                    token_data = await token_response.json()
                                    new_token = token_data.get("access_token")
                                    if new_token:
                                        # System user tokens don't expire
                                        self._save_token_cache(new_token, 31536000)  # 1 year cache
                                        log("Generated System User token successfully!")
                                        return new_token
                                else:
                                    error = await token_response.text()
                                    log(f"System user token generation failed: {error[:200]}")
                        else:
                            log("No system users found for this app")
                    else:
                        error = await response.text()
                        log(f"Failed to get system users: {error[:200]}")
        except Exception as e:
            log(f"System user token error: {e}")

        return None

    async def _exchange_for_long_lived_token(self) -> str | None:
        """Exchange short-lived token for long-lived token"""
        if not META_APP_ID or not META_APP_SECRET:
            log("App ID or App Secret not configured")
            return None

        if not self.access_token:
            log("No access token to exchange")
            return None

        try:
            url = "https://graph.facebook.com/oauth/access_token"
            params = {
                "grant_type": "fb_exchange_token",
                "client_id": META_APP_ID,
                "client_secret": META_APP_SECRET,
                "fb_exchange_token": self.access_token
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        new_token = data.get("access_token")
                        expires_in = data.get("expires_in", 5184000)  # Default 60 days

                        if new_token:
                            self._save_token_cache(new_token, expires_in)
                            log(f"Successfully exchanged for long-lived token (expires in {expires_in // 86400} days)")
                            return new_token
                    else:
                        error = await response.text()
                        log(f"Token exchange failed ({response.status}): {error[:200]}")
        except Exception as e:
            log(f"Token exchange error: {e}")

        return None

    async def debug_token(self) -> dict | None:
        """Debug/inspect current token"""
        if not self.access_token or not META_APP_ID or not META_APP_SECRET:
            return None

        try:
            url = "https://graph.facebook.com/debug_token"
            params = {
                "input_token": self.access_token,
                "access_token": f"{META_APP_ID}|{META_APP_SECRET}"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        token_data = data.get("data", {})

                        is_valid = token_data.get("is_valid", False)
                        expires_at = token_data.get("expires_at", 0)
                        app_id = token_data.get("app_id", "")
                        scopes = token_data.get("scopes", [])

                        if expires_at:
                            expiry = datetime.fromtimestamp(expires_at)
                            log(f"Token valid: {is_valid}, expires: {expiry.strftime('%Y-%m-%d %H:%M')}")
                        else:
                            log(f"Token valid: {is_valid}, never expires (system user token)")

                        return token_data
                    else:
                        error = await response.text()
                        log(f"Debug token failed: {error[:200]}")
        except Exception as e:
            log(f"Debug token error: {e}")

        return None

    async def handle_401_error(self) -> str | None:
        """Called when API returns 401 - generate new token using client_credentials"""
        log("Received 401 Unauthorized - generating new token...")

        # Clear current token validity
        self.token_expiry = None

        # Primary Method: Generate App Access Token using client_credentials grant
        # As per: https://developers.facebook.com/docs/facebook-login/guides/access-tokens/#apptokens
        log("Generating token via client_credentials grant...")
        new_token = await self._generate_app_access_token()
        if new_token:
            os.environ["META_ACCESS_TOKEN"] = new_token
            log(f"New token generated and set: {new_token[:20]}...")
            return new_token

        log("Token generation failed")
        return None


# Global token manager instance
token_manager = TokenManager()


async def get_access_token() -> str:
    """Get a valid Meta access token"""
    return await token_manager.get_valid_token()


async def handle_token_error() -> str | None:
    """Handle 401 error by refreshing token"""
    return await token_manager.handle_401_error()
