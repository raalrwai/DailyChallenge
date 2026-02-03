from typing import Dict, List
from rouge_score import rouge_scorer

class Agent:
    def __init__(self, name: str): self.name = name
    def run(self, state: Dict) -> Dict: raise NotImplementedError

class RetrievalAgent(Agent):
    def __init__(self, name: str, reference: List[str] = None):
        super().__init__(name)
        self.reference = reference or []

    def run(self, state: Dict) -> Dict:
        query = state["query"]
        # minimal retrieval: pretend we got 3 docs
        docs = [f"Doc about {query} #{i}" for i in range(1,4)]

        # -----------------------------
        # Filter docs by similarity to reference using ROUGE
        # -----------------------------
        if self.reference:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            filtered_docs = []
            for doc in docs:
                max_score = max([scorer.score(doc, ref)['rougeL'].fmeasure for ref in self.reference])
                if max_score > 0.3:  # threshold, tweak as needed
                    filtered_docs.append(doc)
            docs = filtered_docs or docs  # fallback to all docs if none pass
        state["retrieved_docs"] = docs
        return state

class ChunkingAgent(Agent):
    def __init__(self, name: str, chunk_size=500, overlap=50):
        super().__init__(name); self.chunk_size=chunk_size; self.overlap=overlap

    def run(self, state: Dict) -> Dict:
        docs = state.get("retrieved_docs", ["Default document"])
        step = self.chunk_size - self.overlap
        state["chunks"] = [doc[i:i+self.chunk_size] for doc in docs for i in range(0,len(doc),step)]
        return state

class ReasoningAgent(Agent):
    def run(self, state: Dict) -> Dict:
        state["answer"] = f"Answer based on {len(state.get('chunks',[]))} chunks"
        return state

class ManagerAgent:
    def __init__(self, model:str, instructions:str, sub_agents:List[Agent]):
        self.sub_agents = sub_agents

    def run(self, query:str) -> Dict:
        state = {"query": query}
        # simple selection logic
        if "retrieve" in query.lower(): chosen="retrieval"
        elif "chunk" in query.lower(): chosen="chunking"
        else: chosen="reasoning"
        agent = next(a for a in self.sub_agents if a.name==chosen)
        return agent.run(state)

if __name__=="__main__":
    reference_docs = ["Document about RAG #1"]  # golden reference for filtering
    agents = [
        RetrievalAgent("retrieval", reference=reference_docs),
        ChunkingAgent("chunking", chunk_size=10, overlap=3),
        ReasoningAgent("reasoning")
    ]
    manager = ManagerAgent("gpt-4o-mini","Decide which agent to run",agents)
    for q in ["retrieve docs about RAG","chunk this document","explain RAG"]:
        print("\nQuery:", q)
        print("State:", manager.run(q))
