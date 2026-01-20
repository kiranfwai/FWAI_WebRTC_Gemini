"""
WhatsApp Business API Client for Voice Calls and Messages
With automatic token refresh on 401 errors
"""

import httpx
from loguru import logger
from typing import Optional, Dict, Any
from config import config

# Import token manager for auto-refresh
try:
    from token_manager import token_manager, get_access_token, handle_token_error
    HAS_TOKEN_MANAGER = True
except ImportError:
    HAS_TOKEN_MANAGER = False
    logger.warning("Token manager not available - token refresh disabled")


class WhatsAppClient:
    """Client for WhatsApp Business API with automatic token refresh"""

    def __init__(self):
        self._update_headers()

    def _update_headers(self):
        """Update headers with current token"""
        token = config.meta_access_token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    async def _refresh_token_and_retry(self) -> bool:
        """Attempt to refresh token after 401 error"""
        if not HAS_TOKEN_MANAGER:
            logger.error("Token manager not available - cannot refresh token")
            return False

        logger.info("Attempting to refresh Meta access token...")
        new_token = await handle_token_error()

        if new_token:
            # Update config and headers
            config.meta_access_token = new_token
            self._update_headers()
            logger.info("Token refreshed successfully!")
            return True

        logger.error("Failed to refresh token")
        return False

    async def make_call(
        self,
        phone_number: str,
        sdp_offer: str,
        ice_candidates: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Initiate an outbound call to a WhatsApp user

        Args:
            phone_number: The phone number to call (with country code, no +)
            sdp_offer: The SDP offer from WebRTC
            ice_candidates: Optional list of ICE candidates

        Returns:
            API response with call_id and other details
        """
        payload = {
            "messaging_product": "whatsapp",
            "to": phone_number,
            "type": "audio",
            "session": {
                "sdp": sdp_offer,
                "sdp_type": "offer"
            }
        }

        if ice_candidates:
            payload["session"]["ice_candidates"] = ice_candidates

        return await self._make_call_with_retry(payload)

    async def _make_call_with_retry(self, payload: dict, retry_count: int = 0) -> Dict[str, Any]:
        """Make call with automatic token refresh on 401"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    config.whatsapp_calls_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"Call initiated successfully: {result}")
                return {"success": True, **result}
        except httpx.HTTPStatusError as e:
            # Handle 401 Unauthorized - try token refresh
            if e.response.status_code == 401 and retry_count == 0:
                logger.warning("Got 401 Unauthorized - attempting token refresh...")
                if await self._refresh_token_and_retry():
                    logger.info("Retrying call after token refresh...")
                    return await self._make_call_with_retry(payload, retry_count=1)

            logger.error(f"HTTP error making call: {e.response.text}")
            return {"success": False, "error": str(e), "details": e.response.text}
        except Exception as e:
            logger.error(f"Error making call: {e}")
            return {"success": False, "error": str(e)}

    async def answer_call(
        self,
        call_id: str,
        sdp_answer: str,
        ice_candidates: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Answer an incoming call with SDP answer

        Args:
            call_id: The call ID from the webhook
            sdp_answer: The SDP answer from WebRTC
            ice_candidates: Optional list of ICE candidates

        Returns:
            API response
        """
        # Accept call directly with SDP answer
        accept_payload = {
            "messaging_product": "whatsapp",
            "call_id": call_id,
            "action": "accept",
            "session": {
                "sdp": sdp_answer,
                "sdp_type": "answer"
            }
        }

        if ice_candidates:
            accept_payload["session"]["ice_candidates"] = ice_candidates

        try:
            async with httpx.AsyncClient() as client:
                logger.info(f"Answering call {call_id} with SDP answer")
                response = await client.post(
                    config.whatsapp_calls_url,
                    headers=self.headers,
                    json=accept_payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"Call answered successfully: {result}")
                return {"success": True, **result}
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error answering call: {e.response.text}")
            return {"success": False, "error": str(e), "details": e.response.text}
        except Exception as e:
            logger.error(f"Error answering call: {e}")
            return {"success": False, "error": str(e)}

    async def send_ice_candidate(
        self,
        call_id: str,
        ice_candidate: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send ICE candidate to WhatsApp

        Args:
            call_id: The call ID
            ice_candidate: The ICE candidate object

        Returns:
            API response
        """
        payload = {
            "call_id": call_id,
            "action": "ice_candidate",
            "ice_candidate": ice_candidate
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    config.whatsapp_calls_url,
                    headers=self.headers,
                    json=payload,
                    timeout=10.0
                )
                return {"success": True}
        except Exception as e:
            logger.error(f"Error sending ICE candidate: {e}")
            return {"success": False, "error": str(e)}

    async def terminate_call(self, call_id: str) -> Dict[str, Any]:
        """
        Terminate an active call

        Args:
            call_id: The call ID to terminate

        Returns:
            API response
        """
        payload = {
            "call_id": call_id,
            "action": "terminate"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    config.whatsapp_calls_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                logger.info(f"Call terminated: {call_id}")
                return {"success": True}
        except Exception as e:
            logger.error(f"Error terminating call: {e}")
            return {"success": False, "error": str(e)}

    async def reject_call(self, call_id: str) -> Dict[str, Any]:
        """
        Reject an incoming call

        Args:
            call_id: The call ID to reject

        Returns:
            API response
        """
        payload = {
            "call_id": call_id,
            "action": "reject"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    config.whatsapp_calls_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                logger.info(f"Call rejected: {call_id}")
                return {"success": True}
        except Exception as e:
            logger.error(f"Error rejecting call: {e}")
            return {"success": False, "error": str(e)}

    async def send_message(
        self,
        phone_number: str,
        message: str
    ) -> Dict[str, Any]:
        """
        Send a text message to a WhatsApp user

        Args:
            phone_number: The phone number to message
            message: The message text

        Returns:
            API response
        """
        payload = {
            "messaging_product": "whatsapp",
            "to": phone_number,
            "type": "text",
            "text": {"body": message}
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    config.whatsapp_messages_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"Message sent to {phone_number}")
                return {"success": True, **result}
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return {"success": False, "error": str(e)}


# Global client instance
whatsapp_client = WhatsAppClient()
