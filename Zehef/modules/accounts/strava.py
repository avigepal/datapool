from lib.Requests import Request
import json


async def strava(target: str):
    """Checks if an email is associated with a Strava account"""
    try:
        from modules.accounts import get_logo 
        params = {"email": target}
        req = await Request("https://www.strava.com/frontend/athletes/email_unique", params=params).get()

        # Ensure response is valid JSON
        try:
            response_json = req.json()
        except json.JSONDecodeError:
            return {"source": "Strava", "status": "error", "message": "Invalid JSON response", "logo": get_logo("Strava")}

        # Strava API returns 'false' when the email is already registered
        if str(response_json).strip().lower() == "false":
            return {"source": "Strava", "status": "found", "logo": get_logo("Strava")}
        else:
            return {"source": "Strava", "status": "not found", "logo": get_logo("Strava")}

    except Exception as e:
        return {"source": "Strava", "status": "error", "message": str(e), "logo": get_logo("Strava")}
