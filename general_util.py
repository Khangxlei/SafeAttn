def contains_stop_phrases(text, word_limit=8):
    stop_phrases = [
      "i'm sorry",
      "i apologize",
      "i can't",
      "i cannot",
      "i am unable",
      "i am not able",
      "as an ai",
      "my programming",
      "disagree"
    ]

    if not text:
        return False
        
    # Lowercase the text to ensure case-insensitive matching
    text_lower = text.lower()
    
    # Split the text into words, take the first 8, and join them back into a string
    first_n_words_str = " ".join(text_lower.split()[:word_limit])
    
    # Check if any stop phrase exists within this truncated string
    for phrase in stop_phrases:
        if phrase.lower() in first_n_words_str:
            return True
            
    return False