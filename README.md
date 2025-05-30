# 🧠 ML + AI Dashboard – "Mini Skynet (Intern Edition)"

Welcome to the wild experiment where Machine Learning meets a Large Language Model... and they both try their best not to embarrass themselves.

## 🚀 What Is This?

This is a humble attempt at building an end-to-end dashboard that:
- Uses **LSTM** and **XGBoost** to predict stuff (sometimes accurately).
- Generates neat-looking graphs so it feels like something important is happening.
- Feeds all that processed model output (in good ol’ JSON) into an **Ollama LLM**.
- Lets you *ask questions* about the results like it’s your personal AI business analyst... minus the expensive coffee habit.

## 🧩 Features

- 📈 Time-series predictions with LSTM and XGBoost
- 🗂️ Preprocessed data + saved graphs (because matplotlib is slow sometimes)
- 🧠 Ollama LLM for "insightful" AI-based commentary
- 💬 Ask questions based on model results like:  
  *“Why does this spike exist?”* or *“Should I panic yet?”*

## 🛠 Tech Stack

- Python 🐍
- LSTM, XGBoost (for model magic)
- Matplotlib / Seaborn (for plot-aesthetic supremacy)
- JSON (for structured confusion)
- Ollama (because GenAI is trendy now)
- Flask (UI framework bullied into working)

## 📦 Folder Structure

```bash
.
├── static/             # static files
├── models/             # Saved ML models
├── graphs/             # All the fancy plots
├── templates/          # The Site
├── uploads/            # The fancy dataset
├── output.json         # Where AI gets its wisdom
├── server.py           # The fancy server
├── model.py            # Tryhard model
└── README.md           # You're reading it now, congrats
```

Known Drawbacks (A.K.A. The “It Works… Kinda” Section)
- UI of the AI Chatbot is... ‘minimalistic’
Think early 2000s HTML with the personality of a toaster. Functional, just not pretty.

- Static Output Dependency
LLM only reads from a saved JSON file. Real-time updates? Yeah, maybe later.

- LLM Insight Accuracy May Vary
It sounds confident but sometimes just vibes with the data instead of analyzing it.

- Hardcoded Model Logic
Adding new models will require… effort (and more coffee).

- Future Possibilities
Actually automate decision-making (not just give suggestions nobody follows).

- Add voice support so it sounds smarter.

- Integrate with Slack to replace the intern.

- Control the coffee machine based on market predictions.

- Clean up the UI so people stop judging it like a MySpace page.

### Disclaimer
This project is for educational purposes only. No dashboards, AI models, or interns were harmed (yet).
Use at your own risk of becoming "the person who built that cool AI thing" at work.
