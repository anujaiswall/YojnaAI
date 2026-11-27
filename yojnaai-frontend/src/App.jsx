import { useState, useRef, useEffect } from "react";
import { Send, Loader2 } from "lucide-react";

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Namaste! I'm SchemeWise. Ask me anything about PM-KISAN, Ayushman Bharat, or PMAY.",
      sources: [],
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage() {
    if (!input.trim() || loading) return;

    const userMsg = { role: "user", text: input, sources: [] };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("http://localhost:3000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMsg.text }),
      });
      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: data.answer, sources: data.sources || [] },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "Couldn't reach the server. Is server.js running on port 3000?",
          sources: [],
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-8 px-4">
      <div className="w-full max-w-2xl flex flex-col h-[85vh] bg-white rounded-2xl shadow-md overflow-hidden">
        <div className="bg-green-700 text-white px-6 py-4">
          <h1 className="text-xl font-semibold">🌾 SchemeWise</h1>
          <p className="text-sm text-green-100">
            Know your rights. Claim what's yours.
          </p>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[80%] rounded-xl px-4 py-2 ${
                  msg.role === "user"
                    ? "bg-green-600 text-white"
                    : "bg-gray-100 text-gray-800"
                }`}
              >
                <p className="whitespace-pre-wrap text-sm">{msg.text}</p>
                {msg.sources.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {[...new Set(msg.sources)].map((src, j) => (
                      <span
                        key={j}
                        className="text-xs bg-green-50 text-green-700 px-2 py-0.5 rounded-full border border-green-200"
                      >
                        📄 {src}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-gray-100 rounded-xl px-4 py-2 flex items-center gap-2 text-sm text-gray-500">
                <Loader2 className="w-4 h-4 animate-spin" />
                Checking official scheme documents...
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        <div className="border-t px-4 py-3 flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about PM-KISAN, Ayushman Bharat, PMAY..."
            className="flex-1 border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500"
          />
          <button
            onClick={sendMessage}
            disabled={loading}
            className="bg-green-700 text-white rounded-lg px-4 py-2 hover:bg-green-800 disabled:opacity-50"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
