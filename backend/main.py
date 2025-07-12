import uvicorn 
from fastapi import FastAPI
from kyoka_engine.models import QueryRequest, DecisionResponse, Justification

# why am i doing this complex thing instead of just using python?
# 1. i love the Go cli 
# 2. Go is fun to learn and work with : ) 

# Why am i setting up a python server instead of just having some script and use Go program then ? 
# 1. Pydantic check 
# 2. Cleaner to do it this way, it would be harsh on me to let the CLI handle all that 
# 3. YES, Microservice Architecture ahh approach 
# 4. Code cleaner =)

# fastapi app
app = FastAPI()

# main endpoint which the Go CLI will call 
@app.post("/check", response_model=DecisionResponse)
def check_query(request: QueryRequest):
    """
    takes a query and returns a structured decision
    """
    print(f"received query: '{request.query}'")

    # fake 
    dummy_response = DecisionResponse(
        decision="approved",
        amount="$50000",
        justification=[
            Justification(
                clause="Section C.4.2: Dummy.",
                reason="hard-coded [testing the connection]"
            )
        ]
    )
    return dummy_response

# run
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=6969)
