import re
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


class RAGEngine:
    DOMAIN_TOPICS = [
        "route", "routing", "delivery", "logistics", "shipment", "freight",
        "itinerary", "stop", "waypoint", "depot", "warehouse", "pickup",
        "dispatch", "fleet", "vehicle", "driver", "trucking", "transport",
        "cargo", "customs", "e-way bill", "manifest", "consignment",
        "last mile", "first mile", "supply chain", "distribution",
        "fuel", "mileage", "navigation", "gps", "tracking",
        "time window", "schedule", "delay", "on-time", "eta", "arrival",
        "temperature", "cold chain", "refrigerated", "hazmat", "dangerous goods",
        "pod", "proof of delivery", "invoice", "bill of lading",
        "kpi", "performance", "efficiency", "cost", "optimization",
        "travel", "distance", "trip", "journey", "road", "highway",
        "rail", "air freight", "sea freight", "port", "airport",
        "tms", "wms", "erp", "telematics", "iot",
        "breakdown", "insurance", "claim", "compliance", "regulation",
        "go from", "going from", "get from", "getting from",
        "travel from", "travelling from", "traveling from",
        "reach", "reaching", "how to reach", "how do i reach",
        "drive from", "driving from", "ride from", "riding from",
        "commute", "commuting",
        "best way", "fastest way", "shortest way", "quickest way",
        "how long", "how far", "how much time",
        "directions", "direction", "navigate", "path from", "path to",
        "way to", "way from", "route from", "route to",
        "from here", "to here", "between",
        "bus", "train", "metro", "cab", "auto", "taxi", "uber", "ola",
        "toll", "highway", "expressway", "flyover", "bridge",
        "traffic", "congestion", "jam", "detour", "bypass",
        "drop", "pick up", "pickup point", "drop off",
        "mumbai", "bombay", "bandra", "kurla", "andheri", "dadar",
        "thane", "navi mumbai", "panvel", "borivali", "kandivali",
        "malad", "goregaon", "jogeshwari", "vile parle", "santacruz",
        "bkc", "nariman", "churchgate", "csmt", "colaba", "worli",
        "lower parel", "prabhadevi", "matunga", "sion", "chembur",
        "ghatkopar", "vikhroli", "kanjurmarg", "bhandup", "mulund",
        "dombivli", "kalyan", "bhiwandi", "vasai", "virar", "mira road",
        "nhava sheva", "jnpt", "nhava",
        "pune", "pimpri", "chinchwad", "hadapsar", "kothrud", "hinjewadi",
        "wakad", "baner", "aundh", "shivajinagar", "talegaon",
        "delhi", "ncr", "gurgaon", "noida", "faridabad", "ghaziabad",
        "bengaluru", "bangalore", "whitefield", "electronic city",
        "hyderabad", "secunderabad", "cyberabad",
        "chennai", "kolkata", "ahmedabad", "surat", "jaipur",
        "lucknow", "chandigarh", "coimbatore", "kochi", "indore",
        "nagpur", "nashik", "aurangabad", "visakhapatnam",
        "zone", "area", "sector", "block", "lane", "street", "nagar",
        "colony", "society", "industrial area", "industrial estate",
        "cargo hub", "logistics park", "cold storage", "godown",
        "weighbridge", "octroi", "rto", "check post", "border",
    ]
    RELEVANCE_THRESHOLD = 0.30

    _NAV_PATTERNS = [
        r"\bfrom\b.{1,60}\bto\b",
        r"\bgo\b.{0,40}\bto\b",
        r"\bget\b.{0,40}\bto\b",
        r"\bread?ch\b",
        r"\bdriv(e|ing)\b.{0,40}\bto\b",
        r"\bhow\b.{0,30}\blong\b",
        r"\bhow\b.{0,30}\bfar\b",
        r"\bdir?ections?\b",
        r"\bnear(est)?\b",
        r"\bway\b.{0,30}\bto\b",
        r"\broute\b.{0,30}\bfrom\b",
        r"\bpath\b.{0,30}\bto\b",
    ]

    def __init__(self, llm, embeddings):
        self.llm          = llm
        self.embeddings   = embeddings
        self._vectordb    = None
        self._build_failed = False   # don't retry a permanently broken build

    def warm_up(self, kb_df):
        """Call once at startup (inside cached AIEngine) to pre-build the vector index."""
        if not kb_df.empty and self.embeddings is not None and self._vectordb is None:
            self._build_vectordb(kb_df)

    def _build_vectordb(self, kb_df):
        if self._vectordb is not None or self.embeddings is None or self._build_failed:
            return
        raw_docs, metadatas = [], []
        for _, row in kb_df.iterrows():
            raw_docs.append(f"[{row['category']} — {row['topic']}]\n{row['content']}")
            metadatas.append({"category": row["category"], "topic": row["topic"]})
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=60, separators=["\n\n", "\n", ". ", " "]
        )
        chunks, chunk_metas = [], []
        for doc, meta in zip(raw_docs, metadatas):
            parts = splitter.split_text(doc)
            chunks.extend(parts)
            chunk_metas.extend([meta] * len(parts))
        try:
            self._vectordb = Chroma.from_texts(
                chunks, self.embeddings, metadatas=chunk_metas, persist_directory="./chroma_kb"
            )
        except Exception:
            self._vectordb  = None
            self._build_failed = True

    def _retrieve(self, question, k=5):
        if self._vectordb is None:
            return [], [], []
        try:
            results = self._vectordb.similarity_search_with_relevance_scores(question, k=k)
            docs, scores, metas = [], [], []
            for doc, score in results:
                docs.append(doc.page_content)
                scores.append(score)
                metas.append(doc.metadata)
            return docs, scores, metas
        except Exception:
            return [], [], []

    def _is_in_domain(self, question, chunks, scores):
        q = question.lower()
        if any(kw in q for kw in self.DOMAIN_TOPICS):
            return True
        for pattern in self._NAV_PATTERNS:
            if re.search(pattern, q):
                return True
        if scores and max(scores) >= self.RELEVANCE_THRESHOLD:
            return True
        return False

    def _keyword_fallback(self, question, kb_df):
        q = question.lower()
        for _, row in kb_df.iterrows():
            words = row["topic"].lower().split() + row["category"].lower().split()
            if any(w in q for w in words):
                return f"**📖 {row['topic']}** *(from {row['category']} KB)*\n\n{row['content']}"
        return (
            "I couldn't find a matching article. "
            "Please ask about routing, delivery, customs, fleet, or other logistics topics."
        )

    def answer(self, question, kb_df):
        from langchain_core.messages import HumanMessage, SystemMessage

        # Build on first answer call only if warm_up wasn't called at startup
        if not kb_df.empty and self.embeddings is not None and self._vectordb is None and not self._build_failed:
            with st.spinner("📚 Indexing knowledge base…"):
                self._build_vectordb(kb_df)

        chunks, scores, metas = self._retrieve(question, k=5)

        if not self._is_in_domain(question, chunks, scores):
            return {
                "text": (
                    "⛔ **Out of scope** — I'm RouteIQ Assistant, specialised exclusively "
                    "in **travel and logistics** topics.\n\n"
                    "I can help with route planning, delivery schedules, customs clearance, "
                    "fleet management, fuel costs, cargo compliance, and related subjects."
                ),
                "sources": [], "grounded": False, "rejected": True,
            }

        if self.llm is None:
            return {"text": self._keyword_fallback(question, kb_df),
                    "sources": [], "grounded": False, "rejected": False}

        relevant = [(c, s, m) for c, s, m in zip(chunks, scores, metas) if s >= self.RELEVANCE_THRESHOLD]

        if relevant:
            context = "\n\n".join(
                f"[Source {i} — {m.get('category','')} / {m.get('topic','')}]\n{c}"
                for i, (c, s, m) in enumerate(relevant, 1)
            )
            system_prompt = (
                "You are RouteIQ Assistant, a logistics and travel planning expert. "
                "Answer ONLY using the provided KB context. "
                "Be concise, precise, and actionable. Use bullet points for lists. "
                "If context doesn't cover the question fully, say so clearly."
            )
            user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
            try:
                resp = self.llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ])
                answer_text = resp.content
            except Exception as e:
                answer_text = self._keyword_fallback(question, kb_df)

            sources = [
                {"category": m.get("category",""), "topic": m.get("topic",""), "score": s}
                for _, s, m in relevant
            ]
            return {"text": answer_text, "sources": sources, "grounded": True, "rejected": False}

        # Broad LLM answer without tight grounding
        system_prompt = (
            "You are RouteIQ Assistant, a logistics and travel expert. "
            "Answer the question using your general knowledge about logistics, "
            "routing, and supply chain. Be helpful and concise."
        )
        try:
            resp = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=question),
            ])
            answer_text = resp.content
        except Exception:
            answer_text = self._keyword_fallback(question, kb_df)

        return {"text": answer_text, "sources": [], "grounded": False, "rejected": False}
