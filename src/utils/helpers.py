import json
import re

def format_response(response):
    """
    Safely format response for JSON display in Streamlit
    """
    if isinstance(response, dict):
        # Already a dictionary, return as-is
        return response
    
    elif isinstance(response, str):
        # Try to parse as JSON first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to fix common issues
            try:
                # Fix missing quotes around property names
                fixed_json = fix_malformed_json(response)
                return json.loads(fixed_json)
            except:
                # If still fails, return as simple dict
                return {
                    "answer": response.strip(),
                    "justification": "Response format could not be parsed as JSON"
                }
    else:
        # Fallback for other types
        return {
            "answer": str(response),
            "justification": "Non-string response converted to string"
        }

def fix_malformed_json(json_string):
    """
    Fix common JSON formatting issues
    """
    # Remove extra whitespace and newlines
    json_string = json_string.strip()
    
    # Fix unquoted property names
    json_string = re.sub(r'(\w+):', r'"\1":', json_string)
    
    # Ensure the string is properly wrapped in braces if missing
    if not json_string.startswith('{'):
        json_string = '{' + json_string
    if not json_string.endswith('}'):
        json_string = json_string + '}'
    
    return json_string
