from lib.Requests import Request
import json
from datetime import datetime

class Cavalier:
    def __init__(self, email: str) -> None:
        self.email = email
        self.api = "https://cavalier.hudsonrock.com/api/json/v2/osint-tools/search-by-email"

    async def loader(self):
        """Fetches email data from HudsonRock API and returns structured JSON"""
        response = await Request(
            self.api, 
            headers={'api-key': 'ROCKHUDSONROCK'}, 
            params={'email': self.email}
        ).get()

        try:
            # Extract 'stealers' data
            stealers_data = response.json().get('stealers', [])

            if stealers_data:
                data = stealers_data[0]

                # Convert date if available
                time_iso = data.get('date_compromised')
                date = None
                if time_iso:
                    t_datetime = datetime.fromisoformat(time_iso.replace("Z", "+00:00"))
                    date = t_datetime.strftime("%Y-%m-%d %H:%M:%S")

                result = {
                    "source": "HudsonRock",
                    "status": "compromised",
                    "total_corporate_services": data.get('total_corporate_services', '/'),
                    "total_user_services": data.get('total_user_services', '/'),
                    "date_compromised": date,
                    "computer_name": data.get('computer_name', '/'),
                    "operating_system": data.get('operating_system', '/'),
                    "ip_address": data.get('ip', '/'),
                    "top_passwords": data.get('top_passwords', []),
                    "top_logins": data.get('top_logins', [])
                }

            else:
                result = {"source": "HudsonRock", "status": "safe"}

        except (KeyError, json.JSONDecodeError):
            result = {"source": "HudsonRock", "status": "error", "message": "Decode error"}

        return result  # Return JSON response
