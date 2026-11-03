import os
import sqlite3
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CUSTOMERS_DB = {
    1: {"name": "Alice", "plan": "Basic", "email": "alice@test.com"},
    2: {"name": "Bob", "plan": "Premium", "email": "bob@test.com"},
    3: {"name": "Charlie", "plan": "Enterprise", "email": "charlie@test.com"}
}

def setup_orders_db():
    conn = sqlite3.connect("orders.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            status TEXT,
            tracking_info TEXT
        )
    """)
    cursor.execute("DELETE FROM orders")
    cursor.executemany("INSERT INTO orders VALUES (?, ?, ?)", [
        (1041, "Shipped", "In transit - expected tomorrow"),
        (1042, "Out for delivery", "Driver assigned, arriving by 5 PM"),
        (1043, "Processing", "Warehouse packing order")
    ])
    conn.commit()
    conn.close()

def get_customer(customer_id: int) -> str:
    """Retrieve customer profile and subscription plan by customer ID."""
    cust = CUSTOMERS_DB.get(int(customer_id))
    if cust:
        return json.dumps(cust)
    return json.dumps({"error": f"Customer ID {customer_id} not found."})

def get_order_status(order_id: int) -> str:
    """Retrieve real-time order status and shipment updates by order ID."""
    conn = sqlite3.connect("orders.db")
    cursor = conn.cursor()
    cursor.execute("SELECT status, tracking_info FROM orders WHERE order_id = ?", (int(order_id),))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.dumps({"order_id": order_id, "status": row[0], "details": row[1]})
    return json.dumps({"error": f"Order ID {order_id} not found."})

TOOL_MAP = {
    "get_customer": get_customer,
    "get_order_status": get_order_status
}

def run_tool_calling_agent(user_query: str):
    print(f"\nUser Query: '{user_query}'")
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=user_query,
        config=types.GenerateContentConfig(
            tools=[get_customer, get_order_status],
            temperature=0
        )
    )

    function_calls = response.function_calls
    if not function_calls:
        print("Answer:", response.text)
        return

    tool_trace = " | ".join([f"Tool: {fc.name}({fc.args.get('customer_id') or fc.args.get('order_id')})" for fc in function_calls])
    print(tool_trace)

    tool_response_parts = []
    for fc in function_calls:
        func_name = fc.name
        func_args = fc.args
        if func_name not in TOOL_MAP:
            tool_result = {"error": f"Tool '{func_name}' is not recognized."}
        else:
            func = TOOL_MAP[func_name]
            try:
                raw_res = func(**func_args)
                tool_result = json.loads(raw_res)
            except Exception as e:
                tool_result = {"error": str(e)}

        tool_response_parts.append(
            types.Part.from_function_response(
                name=func_name,
                response={"result": tool_result}
            )
        )

    final_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text=user_query)]),
            response.candidates[0].content,
            types.Content(role="tool", parts=tool_response_parts)
        ]
    )

    print(f"Answer: {final_response.text.strip()}")

if __name__ == "__main__":
    setup_orders_db()
    run_tool_calling_agent("Which plan is customer 2 on and where is order 1042?")
    run_tool_calling_agent("What is the status of order 9999 for customer 88?")