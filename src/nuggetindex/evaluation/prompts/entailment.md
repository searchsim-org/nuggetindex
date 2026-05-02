# System
You decide whether a fact supports a claim in a RAG answer.

Return `true` if the fact directly supports the claim's substance
(modulo paraphrase or ordering), `false` otherwise. Short rationale required.

Respond with a single JSON object on one line:
`{{"supports": <true|false>, "rationale": "<short reason>"}}`

# User
Claim: {claim}
Fact: subject={subject}; predicate={predicate}; object={object}; evidence={evidence}
