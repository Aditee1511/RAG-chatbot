# RAG Chatbot — Project Overview

A simple explanation of what this project does, what data it uses, how it
works, and how to run it.

## 1. Objective

Build a chatbot that can answer questions from a large set of PDF documents.

Instead of guessing or using general knowledge, the chatbot reads the
actual PDFs, finds the most relevant parts, and answers using only that
information. Every answer also shows **which PDF and which page** the
answer came from, so the user can verify it.

The target: answer most questions in **2 to 5 seconds**, using only
**free and open-source tools** (no paid software required).

## 2. Dataset — What we used, and why

We used **10 real PDF documents published by the Government of India**,
related to education and skill development:

- National Curriculum Framework 2023 (NCERT)
- Economic Survey 2023-24 (Ministry of Finance)
- UGC Annual Reports (3 different years)
- AISHE Report (higher education statistics)
- MSDE Annual Reports (2 different years — skill development)
- NITI Aayog report on school education
- AICTE Approval Process Handbook

**Why this dataset?**
This challenge is similar to what an organization like **NavGurukul** would
need — NavGurukul trains students from low-income backgrounds for tech
jobs, and its work connects directly to Indian education and skilling
policy. So instead of using random internet content, we picked real,
official documents from this exact space. All 10 documents are genuine,
publicly available government PDFs, each more than 200 pages long.

## 3. How it works (the flow)

### Step A — Loading the documents (done once)

1. **Read the PDF** — extract the text from each page.
2. **Handle scanned pages** — some pages are images/scanned tables with no
   readable text. For those, we use OCR (Optical Character Recognition) to
   read the text from the image.
3. **Clean the text** — remove repeated headers/footers, detect the
   language of the text.
4. **Break into chunks** — split each document into small pieces (about
   800 words each), so the system can search through them efficiently.
   Each chunk remembers which PDF and which page it came from.
5. **Convert to numbers (embeddings)** — each chunk is converted into a
   list of numbers that represents its meaning. This lets the computer
   compare "meaning" instead of just matching exact words.
6. **Store in a database** — all these chunks and their numbers are saved
   in a vector database (LanceDB), built for fast searching.

### Step B — Answering a question (happens every time a user asks something)

1. The user types a question in the chat.
2. The question is also converted into numbers (same method as above).
3. The database quickly finds the 20 chunks whose meaning is closest to
   the question.
4. A second, more careful model re-checks those 20 chunks and picks the
   best 3.
5. Those 3 chunks are sent to an AI model (the LLM), which writes a final
   answer using only that information — and adds the source (PDF name +
   page number) to each claim.
6. The answer, sources, and how long it took are shown to the user.

## 4. What we used (tools and why)

| Purpose | Tool | Why we chose it |
|---|---|---|
| Read PDF text | PyMuPDF | Fast, reliable, free |
| Read scanned pages | RapidOCR | Free, no extra software needed to install |
| Convert text to numbers | multilingual-e5-small | Free, open-source, works for English and Indian languages |
| Store & search the numbers | LanceDB | Free, no server/database setup needed |
| Re-check top results | Cross-encoder reranker | Improves accuracy of the final 3 chunks picked |
| Write the final answer | Groq (running an open-source AI model) | Free and extremely fast — this is what makes the 2-5 second target possible |
| Chat screen | Streamlit | Simple way to build a clean web chat interface |

## 5. How to run this project

```bash
# 1. Create a Python environment and install everything needed
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Add your free Groq API key
copy .env.example .env
# open .env and paste your key (get one free at console.groq.com)

# 3. Download the 10 PDFs
python scripts/download_corpus.py

# 4. Process the PDFs (reads, chunks, converts to numbers, saves to database)
#    This takes about 15-20 minutes and only needs to be done once.
python scripts/ingest.py

# 5. Start the chatbot
streamlit run app/app.py
```

Then open **http://localhost:8501** in your browser and start asking
questions.

## 6. How we check it's working well

We built a small testing tool that:
- Automatically creates sample questions from the real PDF content
- Asks the chatbot those questions
- Checks: Did it find the right page? Was the answer fast? Did it cite
  the source correctly?

Run it with:
```bash
python eval/generate_eval_set.py
python eval/evaluate.py
```

This gives us real numbers (not guesses) on accuracy and speed, saved in
`eval/eval_report.json`.
