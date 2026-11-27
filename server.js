import { GoogleGenAI } from "@google/genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { pipeline } from "@xenova/transformers";
import express from "express";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });

// ---------- LOCAL EMBEDDING MODEL ----------
let embedder = null;

async function getEmbedder() {
  if (!embedder) {
    console.log("Loading local embedding model...");
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
    console.log("Model ready.");
  }
  return embedder;
}

async function getEmbedding(text) {
  console.log("→ computing local embedding...");
  const model = await getEmbedder();
  const output = await model(text, { pooling: "mean", normalize: true });
  console.log("→ embedding computed, length:", output.data.length);
  return Array.from(output.data);
}

// ---------- QUERY REWRITING ----------
async function rewriteQuery(userQuery) {
  const prompt = `You are a query rewriting assistant for Yojna-AI, a system that helps Indians find government scheme information.

Rewrite the user's query to be clearer for semantic search:
- Fix typos and expand abbreviations (e.g. "pm-kishan" → "PM-KISAN")
- Convert colloquial phrasing into terms likely to appear in official government documents
- Keep it short — one sentence, no explanation
- If the query is already clear, return it unchanged

Examples:
"om-kisan eligibility" → "PM-KISAN scheme eligibility criteria"
"I lost my farming land, what help can I get" → "government compensation or support schemes for farmers who lost agricultural land"
"house scheme for poor" → "PMAY housing scheme for economically weaker sections"

User query: "${userQuery}"

Rewritten query:`;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
    });
    const rewritten = response.text.trim();
    console.log(`Query rewritten: "${userQuery}" → "${rewritten}"`);
    return rewritten;
  } catch (err) {
    console.log("Query rewriting failed, using original query:", err.message);
    return userQuery; // fallback — never let this break the pipeline
  }
}

// ---------- RETRIEVE CONTEXT ----------
async function retrieveContext(originalQuery) {
  const index = pinecone.index(process.env.PINECONE_INDEX);

  const searchQuery = await rewriteQuery(originalQuery);
  const queryEmbedding = await getEmbedding(searchQuery);

  console.log("→ calling Pinecone query...");
  const results = await index.query({
    vector: queryEmbedding,
    topK: 8, // pull a few more candidates now, since we'll filter/rerank next
    includeMetadata: true,
  });
  console.log("→ got Pinecone results, matches:", results.matches?.length);

  if (!results.matches) return [];

  const relevant = results.matches.filter((m) => m.score > 0.3); // slightly looser, reranker will clean up

  return relevant.map((match) => ({
    text: match.metadata.text,
    source: match.metadata.source,
    score: match.score,
  }));
}



// ---------- GENERATE ANSWER (still Gemini — generous quota) ----------
async function generateAnswer(query, context) {
  const contextText = context
    .map((c, i) => `[Source ${i + 1}: ${c.source}]\n${c.text}`)
    .join("\n\n");

  const prompt = `
You are SchemeWise, an AI assistant that helps users understand Indian Government Schemes.

STRICT RULES:
- ONLY answer from the supplied context.
- If the answer is not present, reply:
"I don't have information about this in my knowledge base. Please visit the official government website."
- Never hallucinate.
- Mention the source.
- Keep the answer simple.

CONTEXT:
${contextText}

QUESTION:
${query}
`;

  console.log("→ calling Gemini generateContent...");
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: prompt,
  });
  console.log("→ got generateContent response");

  return response.text;
}

// ---------- CHAT API ----------
app.get("/", (req, res) => {
  console.log("Health route hit");
  res.send("Yojnaai- server is running!");
});
app.post("/api/chat", async (req, res) => {
  try {
    const { query } = req.body;

    if (!query) {
      return res.status(400).json({ error: "Query is required" });
    }

    console.log("Question:", query);

    const context = await retrieveContext(query);
    console.log("Retrieved:", context.length);

    if (context.length === 0) {
      return res.json({
        answer:
          "I don't have information about this in my knowledge base. Please visit the official government website.",
        sources: [],
      });
    }

    const answer = await generateAnswer(query, context);

    res.json({
      answer,
      sources: context.map((c) => c.source),
    });
  } catch (err) {
    console.error("ERROR:", err);
    res.status(500).json({ error: err.message });
  }
});

// ---------- START ----------
app.listen(3000, () => {
  console.log("🚀 SchemeWise server running on port 3000");
});
