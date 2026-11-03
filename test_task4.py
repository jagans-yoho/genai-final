import requests

API_URL = "http://127.0.0.1:8000/ask"

test_questions = [
    {"q": "What protocols are supported for auto-backup scheduling?", "type": "Answerable"},
    {"q": "Where can I configure backup settings in CloudSync?", "type": "Answerable"},
    {"q": "How can I reset administrative 2FA?", "type": "Answerable"},
    {"q": "What is the policy regarding SMS OTP sharing?", "type": "Answerable"},
    {"q": "What is the refund turnaround time for annual subscriptions?", "type": "Answerable"},
    {"q": "Within how many days must an annual plan be cancelled to claim refund?", "type": "Answerable"},
    {"q": "What is the secret recipe for company cafeteria coffee?", "type": "Unanswerable"},
    {"q": "Can I install CloudSync on Nintendo Switch?", "type": "Unanswerable"},
    {"q": "Who won the 2024 ICC Cricket World Cup?", "type": "Unanswerable"},
    {"q": "What discount code gives 90% off lifetime access?", "type": "Unanswerable"}
]

print(" Running 10 RAG Groundedness Tests ")
for t in test_questions:
    response = requests.post(API_URL, json={"question": t["q"]})
    print(f"\n[{t['type']}] Q: {t['q']}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Ans: {data.get('answer')}")
        print(f"Sources: {data.get('sources')}")
    else:
        print(f"Error Response: {response.text}")