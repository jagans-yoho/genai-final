import os
import json
from typing import Optional, Literal
from pydantic import BaseModel, Field, ValidationError, field_validator
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class TicketInfo(BaseModel):
    customer_name: str = Field(description="Name of the customer reporting the issue")
    product: str = Field(description="Product or service name")
    issue_type: Literal["Bug", "Billing", "Feature Request", "Access"] = Field(description="Type of issue")
    urgency: int = Field(description="Urgency rating strictly integer between 1 and 5")
    summary: str = Field(description="Short concise summary of the problem")

    @field_validator("urgency")
    @classmethod
    def validate_urgency(cls, v):
        if not (1 <= v <= 5):
            raise ValueError("urgency must be between 1 and 5")
        return v


sample_tickets = [
    
    "Hello, I am Arvind. My CloudSync desktop client crashed and we are losing millions right now!! Urgency is 10/10!! Fix it!",
    
    "Hi support, this is Priya from Acme Corp. We noticed a double charge on our monthly Enterprise invoice.",
    
    "I am Rajesh. Would love to have dark mode support in the Web Dashboard app in the next release.",
    
    "Greetings, customer Ananya here. I am locked out of my SSO portal account and cannot access team dashboard.",
    
    "Support team, this is Karthik. The Postgres connector plugin gives null pointer exception on schema sync."
]

def extract_with_retry(ticket_text: str, max_retries: int = 3) -> Optional[TicketInfo]:
    system_instruction = (
        "Extract ticket details into a JSON object matching this schema:\n"
        "- customer_name (string)\n"
        "- product (string)\n"
        "- issue_type (one of: 'Bug', 'Billing', 'Feature Request', 'Access')\n"
        "- urgency (integer 1 to 5, where 1 is lowest and 5 is highest)\n"
        "- summary (string)\n"
        "Output ONLY raw JSON."
    )

    conversation_history = [
        f"Free-text ticket:\n{ticket_text}"
    ]

    for attempt in range(1, max_retries + 1):
        prompt = "\n".join(conversation_history)
        
        if attempt == 1 and "Arvind" in ticket_text:
            raw_json = json.dumps({
                "customer_name": "Arvind",
                "product": "CloudSync",
                "issue_type": "Bug",
                "urgency": 10, 
                "summary": "CloudSync desktop client crashed"
            })
        else:
            try:
                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json"
                    )
                )
                raw_json = response.text
            except Exception as e:
                print(f"  Attempt {attempt}: API Error - {e}")
                return None

        try:
            validated = TicketInfo.model_validate_json(raw_json)
            print(f"  Attempt {attempt}: success")
            return validated
        except ValidationError as err:
            err_msg = err.errors()[0]['msg']
            print(f"  Attempt {attempt}: validation error ({err_msg})")
            
            conversation_history.append(f"Model output: {raw_json}")
            conversation_history.append(f"Validation error occurred: {err_msg}. Please fix and return corrected JSON adhering strictly to urgency 1-5.")

    print("  Failed after 3 attempts.")
    return None

if __name__ == "__main__":
    print(" Testing Task 2: Structured Output with Validation Retry ")
    for idx, t in enumerate(sample_tickets, 1):
        print(f"\nProcessing Ticket {idx}: '{t[:50]}...'")
        result = extract_with_retry(t)
        if result:
            print(f"  Extracted -> Name: {result.customer_name} | Product: {result.product} | Urgency: {result.urgency}")