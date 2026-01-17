from lib.Requests import Request
import hashlib


async def gravatar(target: str):
    """Checks if an email is associated with a Gravatar account"""
    try:
        from modules.accounts import get_logo 
        # Generate MD5 hash of the email (Gravatar uses lowercase & stripped email hashes)
        encoded_email = target.lower().strip().encode('utf-8')
        hashed_email = hashlib.md5(encoded_email).hexdigest()

        # Gravatar Profile API
        r = await Request(f"https://en.gravatar.com/{hashed_email}.json").get()

        # If the response contains "User not found", return not found
        if "User not found" in r.text:
            return {"source": "Gravatar", "status": "not found", "logo": get_logo("Gravatar")}

        # Extract user data
        data = r.json().get('entry', [])[0]

        if data:
            username = data.get('displayName', 'N/A')
            avatar_url = data.get('thumbnailUrl', 'N/A').replace("\\", "")
            profile_url = f"https://gravatar.com/{hashed_email}"  # Gravatar profile link by hash

            return {
                "source": "Gravatar",
                "status": "found",
                "username": username,
                "avatar_url": avatar_url,
                "profile_url": profile_url,
                "logo": get_logo("Gravatar")
            }

        return {"source": "Gravatar", "status": "not found", "logo": get_logo("Gravatar")}

    except Exception as e:
        return {"source": "Gravatar", "status": "error", "message": str(e), "logo": get_logo("Gravatar")}
