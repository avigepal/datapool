import secrets
import base64

key = secrets.token_bytes(8)  # Generates a 32-byte (256-bit) key
key_base64 = base64.b64encode(key).decode('utf-8')  # Encode to base64 for storage
print(key_base64)