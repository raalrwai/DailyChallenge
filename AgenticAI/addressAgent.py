from AgenticAI import Agent, FunctionTool, Runner

@function_tool
def getAddress():
    return "1156 address St."

def getSquareFeet():
    return "1156 address St."

house = Agent{
    name =  "Address Agent",
    instructions = "You're a get Address Agent. Your main responsibility is ", 
    "to allow a user to get their address"
    tools = [getAddress, getSquareFeet]
}

if "__name__" = "__main__":  
    address = Runner.Run(house.getAddress)
    print(f"Your address is {address}")