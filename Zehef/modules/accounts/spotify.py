from lib.Requests import Request
import json


async def spotify(target: str):
    """Checks if an email is associated with a Spotify account"""
    try:
        from modules.accounts import get_logo 
        url = f"https://spclient.wg.spotify.com/signup/public/v1/account?validate=1&email={target}"
        r = await Request(url).get()

        try:
            response_json = r.json()
        except json.JSONDecodeError:
            return {"source": "Spotify", "status": "error", "message": "Invalid JSON response", "logo": get_logo("Spotify")}

        if response_json.get("status") == 20:
            return {"source": "Spotify", "status": "found", "logo": get_logo("Spotify")}
        else:
            return {"source": "Spotify", "status": "not found", "logo": get_logo("Spotify")}

    except Exception as e:
        return {"source": "Spotify", "status": "error", "message": str(e), "logo": get_logo("Spotify")}
