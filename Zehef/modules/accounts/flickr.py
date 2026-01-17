import re
from lib.Requests import Request


async def flickr(target: str):
    """Checks if an email or username is associated with a Flickr account"""
    try:
        from modules.accounts import get_logo 
        # Step 1: Fetch API Key from Flickr Homepage
        r = await Request("https://www.flickr.com/").get()
        key_pattern = r'[a-f0-9]{32}'
        keys = re.findall(key_pattern, r.text)
        api_keys = set(keys)

        if not api_keys:
            return {
                "source": "Flickr",
                "status": "error",
                "message": "No API keys found",
                "logo": get_logo("Flickr")
            }

        # Step 2: Try different API keys until we get a result
        for key in api_keys:
            api_url = "https://api.flickr.com/services/rest"
            params = {
                'username': target,
                'exact': 0,
                'extras': 'path_alias%2Crev_ignored%2Crev_contacts%2Cis_pro%2Cicon_urls%2Clocation%2Crev_contact_count%2Cuse_vespa%2Cdate_joined',
                'per_page': 5,
                'page': 0,
                'show_more': 1,
                'perPage': 50,
                'loadFullContact': 1,
                'viewerNSID': None,
                'method': 'flickr.people.search',
                'api_key': key,
                'format': 'json',
                'hermes': 1,
                'hermesClient': 1,
                'nojsoncallback': 1
            }

            r = await Request(api_url, params=params).get()

            try:
                data = r.json().get('people', {}).get('person', [])[0]

                if data:
                    return {
                        "source": "Flickr",
                        "status": "found",
                        "username": data.get('username'),
                        "realname": data.get('realname', 'N/A'),
                        "user_id": data.get('dbid', 'N/A'),
                        "profile_url": f"https://www.flickr.com/people/{data.get('nsid')}/",
                        "logo": get_logo("Flickr")
                    }
            except (IndexError, KeyError):
                continue  # Try next API key if available

        return {"source": "Flickr", "status": "not found", "logo": get_logo("Flickr")}

    except Exception as e:
        return {
            "source": "Flickr",
            "status": "error",
            "message": str(e),
            "logo": get_logo("Flickr")
        }
