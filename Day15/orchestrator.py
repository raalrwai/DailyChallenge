from typing import Dict, List

class Agent:
    def __init__(self, name: str): self.name = name
    def run(self, state: Dict) -> Dict: raise NotImplementedError

class RetrievalAgent(Agent):
    def run(self, state: Dict) -> Dict:
        query = state.get("query","")
        state["retrieved_docs"] = [f"Doc about {query} #{i}" for i in range(1,4)]
        print(f"[{self.name}] Retrieved {len(state['retrieved_docs'])} docs")
        return state

class ChunkingAgent(Agent):
    def __init__(self, name: str, chunk_size=500, overlap=50):
        super().__init__(name); self.chunk_size=chunk_size; self.overlap=overlap
    def run(self, state: Dict) -> Dict:
        chunks=[]
        step=self.chunk_size-self.overlap
        for doc in state.get("retrieved_docs",[]):
            chunks.extend([doc[i:i+self.chunk_size] for i in range(0,len(doc),step)])
        state["chunks"]=chunks
        print(f"[{self.name}] Created {len(chunks)} chunks")
        return state

class ReasoningAgent(Agent):
    def run(self, state: Dict) -> Dict:
        chunks=state.get("chunks",[])
        state["answer"]=f"Based on {len(chunks)} chunks, the answer is XYZ."
        print(f"[{self.name}] Generated reasoning answer")
        return state

class ManagerAgent:
    def __init__(self, model:str, instructions:str, sub_agents:List[Agent]):
        self.model,self.instructions,self.sub_agents = model,instructions,sub_agents
    def run(self, query:str) -> Dict:
        state={"query":query}
        if "retrieve" in query.lower(): chosen="retrieval"
        elif "chunk" in query.lower(): chosen="chunking"
        else: chosen="reasoning"
        agent=next(a for a in self.sub_agents if a.name==chosen)
        print(f"[Manager] Running agent: {agent.name}")
        return agent.run(state)

if __name__=="__main__":
    sub_agents=[
        RetrievalAgent("retrieval"),
        ChunkingAgent("chunking", chunk_size=10, overlap=3),
        ReasoningAgent("reasoning")
    ]
    manager=ManagerAgent("gpt-4o-mini","Decide which agent to run",sub_agents)
    queries=["retrieve documents about RAG","chunk this document","explain what RAG is"]
    for q in queries:
        print("\nUser Query:",q)
        final_state=manager.run(q)
        print("Final State:",final_state)
