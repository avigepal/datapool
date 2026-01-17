from lib.Requests import Request
from datetime import datetime


async def github(target: str):
    """Checks if an email is associated with a GitHub account"""
    try:
        from modules.accounts import get_logo 
        # Search for the user by email
        r = await Request(f"https://api.github.com/search/users?q={target}+in:email").get()
        response_json = r.json()

        if response_json.get("total_count", 0) == 0:
            return {"source": "GitHub", "status": "not found"}

        try:
            user_data = response_json['items'][0]
            username = user_data.get('login')

            # Fetch user profile details
            api = await Request(f"https://api.github.com/users/{username}").get()
            profile_data = api.json()

            # Parse creation & update timestamps
            creation = profile_data.get("created_at")
            update = profile_data.get("updated_at")

            c_date = datetime.fromisoformat(creation.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S") if creation else "N/A"
            u_date = datetime.fromisoformat(update.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S") if update else "N/A"

            return {
                "source": "GitHub",
                "status": "found",
                "username": username,
                "name": profile_data.get("name", "N/A"),
                "user_id": user_data.get("id", "N/A"),
                "avatar_url": user_data.get("avatar_url", "N/A"),
                "created_on": c_date,
                "updated_on": u_date,
                "profile_url": f"https://github.com/{username}",
                "logo": get_logo("GitHub"),
            }

        except (KeyError, IndexError):
            return {"source": "GitHub", "status": "error", "message": "User data parsing failed", "logo": get_logo("GitHub")}
    except Exception as e:
        return {"source": "GitHub", "status": "error", "message": str(e), "logo": get_logo("GitHub")}
