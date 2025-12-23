MODEL_CAPABILITIES = {
    # -----------------
    # Short-context
    # -----------------
    "sshleifer/distilbart-cnn-12-6": {"type": "short"},
    "t5-small": {"type": "short"},
    "facebook/bart-large-cnn": {"type": "short"},
    "google/flan-t5-base": {"type": "short"},
    "google/pegasus-xsum": {"type": "short"},
    "facebook/bart-base": {"type": "short"},

    # -----------------
    # Long-context
    # -----------------
   "gemini-2.5-flash": {
        "type": "long",
        "provider": "gemini"
    }
}