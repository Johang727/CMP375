Video:

- Four Agents
- One orchistrator, etc etc

Message Flow:

- Channel monitor
- Routing engine
- Gateway
- Agent Loop
- Auto-reply
- Channel Sender
- Channel monitor

GPT & LLMs

- Generative ive Pre-Trained Transformer

Chat is the interface, GPT is the Pre-Trained

Generative:

- Generates the next token

Pre-trained:

- Books
- Websites
- Learns patters within billions of these.

Transformer: Specific design for how an AI system processes language.

- Ability to look at all the words at once & how they relate to each other. Earlier AI systems couldn't see the relationship between all the words.
- Chatbots
- Spamfilters

LLM: Large Language model

Large:

- Lots of training data & parameters
- Modern LLMs are trained on trillions of words

Learning

The model is shown all this text and modifies the parameters it uses to predict the next word.

- After billions of iterations, the agent learns the next token accurately.

GPT-2 had 1.5 B params, GPT-3 175 B, cost increases per param.

Can't do math reliably, know recent events

Hallucination: Making up confident-sounding answers without knowing the facts.

Every API agent uses an API.

Token is a chunk of text, the currency.

- Common words (in the training data) are not split.
- Token-ization
- One token typically equals 3-4 token

Output tokens are 4x more than an input token.

Context windows increase per newer model. Llama 4 Scout has 10M token context window.

LLMs have randomness in their token. Temperature is the way this is chosen.

- Temp 0 = Always the same
- Temp 2 = Always different and can lose it :p

Data extraction: 0
Code generation: 0 - 0.2
Conversation: 0.5 - 0.7
Creative writing: 0.8 - 1.2

Context Window: Hard token limit on what the model can see.

If the context window is exceeded:

- Truncation: the older messages are dropped
- Summarization: The older messages are summarized

Agents needs to hold everything in context. More tokens = higher cost, but better results.
