import re
from bson import ObjectId, Binary, Decimal128
from datetime import datetime

# Dictionary of country dialing codes
COUNTRY_DIALING_CODES = {
    "93": "Afghanistan", "355": "Albania", "213": "Algeria", "376": "Andorra", "244": "Angola", "54": "Argentina",
    "374": "Armenia", "61": "Australia", "43": "Austria", "994": "Azerbaijan", "1": "Bahamas", "973": "Bahrain", "880": "Bangladesh",
    "375": "Belarus", "32": "Belgium", "501": "Belize", "229": "Benin", "975": "Bhutan", "591": "Bolivia", "387": "Bosnia and Herzegovina",
    "267": "Botswana", "55": "Brazil", "359": "Bulgaria", "226": "Burkina Faso", "257": "Burundi", "855": "Cambodia", "237": "Cameroon",
    "1": "Canada", "235": "Chad", "56": "Chile", "86": "China", "57": "Colombia", "269": "Comoros", "506": "Costa Rica", "385": "Croatia",
    "53": "Cuba", "357": "Cyprus", "420": "Czech Republic", "45": "Denmark", "1": "Dominican Republic", "593": "Ecuador", "20": "Egypt",
    "503": "El Salvador", "372": "Estonia", "251": "Ethiopia", "358": "Finland", "33": "France", "995": "Georgia", "49": "Germany",
    "233": "Ghana", "30": "Greece", "502": "Guatemala", "504": "Honduras", "852": "Hong Kong", "36": "Hungary", "354": "Iceland", "91": "India",
    "62": "Indonesia", "98": "Iran", "964": "Iraq", "353": "Ireland", "972": "Israel", "39": "Italy", "1": "Jamaica", "81": "Japan",
    "962": "Jordan", "7": "Kazakhstan", "254": "Kenya", "965": "Kuwait", "856": "Laos", "371": "Latvia", "961": "Lebanon", "218": "Libya",
    "370": "Lithuania", "352": "Luxembourg", "853": "Macau", "60": "Malaysia", "960": "Maldives", "223": "Mali", "356": "Malta", "52": "Mexico",
    "373": "Moldova", "377": "Monaco", "976": "Mongolia", "212": "Morocco", "95": "Myanmar", "977": "Nepal", "31": "Netherlands", "64": "New Zealand",
    "505": "Nicaragua", "234": "Nigeria", "850": "North Korea", "47": "Norway", "968": "Oman", "92": "Pakistan", "507": "Panama", "595": "Paraguay",
    "51": "Peru", "63": "Philippines", "48": "Poland", "351": "Portugal", "974": "Qatar", "40": "Romania", "7": "Russia", "250": "Rwanda",
    "966": "Saudi Arabia", "221": "Senegal", "381": "Serbia", "65": "Singapore", "421": "Slovakia", "386": "Slovenia", "27": "South Africa",
    "82": "South Korea", "34": "Spain", "94": "Sri Lanka", "249": "Sudan", "46": "Sweden", "41": "Switzerland", "963": "Syria", "886": "Taiwan",
    "992": "Tajikistan", "255": "Tanzania", "66": "Thailand", "216": "Tunisia", "90": "Turkey", "993": "Turkmenistan", "256": "Uganda",
    "380": "Ukraine", "971": "United Arab Emirates", "44": "United Kingdom", "1": "United States", "598": "Uruguay", "998": "Uzbekistan",
    "58": "Venezuela", "84": "Vietnam", "967": "Yemen", "260": "Zambia", "263": "Zimbabwe"
}

def normalize_phone_number(phone: str):
    phone = re.sub(r"\D", "", phone)  # Remove all non-numeric characters
    for code in sorted(COUNTRY_DIALING_CODES.keys(), key=len, reverse=True):  # Sort by length to match longest code first
        if phone.startswith(code):
            return phone, phone[len(code):]  # Full number, Local number without country code
    return phone, phone

def clean_mongo_data(doc):
    """Recursively convert MongoDB fields to JSON-safe formats."""
    if isinstance(doc, dict):
        return {key: clean_mongo_data(value) for key, value in doc.items()}
    elif isinstance(doc, list):
        return [clean_mongo_data(item) for item in doc]
    elif isinstance(doc, ObjectId):
        return str(doc)  # Convert ObjectId to string
    elif isinstance(doc, datetime):
        return doc.isoformat()  # Convert datetime to ISO format
    elif isinstance(doc, Binary):
        return doc.decode("utf-8", errors="ignore")  # Convert Binary to string
    elif isinstance(doc, Decimal128):
        return float(doc.to_decimal())  # Convert Decimal128 to float
    elif isinstance(doc, int):  # Handle large MongoDB Int64 values safely
        return int(doc)
    return doc