from lib.Requests import Request


async def bandlab(target: str):
    """Checks if an email is associated with a BandLab account"""
    try:
        from modules.accounts import get_logo 
        r = await Request(
            "https://www.bandlab.com/api/v1.3/validation/user",
            params={'email': target}
        ).get()

        response_json = r.json()
        logo = get_logo('BandLab')  # Fetch BandLab logo path

        if response_json.get('isValid') and response_json.get('isAvailable') is False:
            return {"source": "BandLab", "status": "found", "logo": logo}
        else:
            return {"source": "BandLab", "status": "not found", "logo": logo}

    except Exception as e:
        return {"source": "BandLab", "status": "error", "message": str(e), "logo": get_logo("BandLab")}
