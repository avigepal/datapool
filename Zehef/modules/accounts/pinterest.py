import json
from lib.Requests import Request

async def pinterest(target: str):
    """Checks if an email is associated with a Pinterest account"""
    try:
        from modules.accounts import get_logo  # Import get_logo function
        params = {
            "source_url": "/",
            "data": json.dumps({"options": {"email": target}, "context": {}})
        }

        r = await Request("https://www.pinterest.fr/resource/EmailExistsResource/get/", params=params).get()
        response_json = r.json()
        logo = get_logo("Pinterest")

        # Check if the response contains account existence data
        if response_json.get("resource_response", {}).get("data"):
            return {"source": "Pinterest", "status": "found", "logo": logo}
        else:
            return {"source": "Pinterest", "status": "not found", "logo": logo}

    except Exception as e:
        return {"source": "Pinterest", "status": "error", "message": str(e), "logo": get_logo("Pinterest")}
