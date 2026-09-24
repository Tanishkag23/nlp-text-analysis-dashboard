"""Short passages used by the 'Load sample' control in the sidebar.

Each one is chosen to exercise a different part of the pipeline: named
entities, inflected verb forms, numbers and dates, and long function-word
chains for stopword removal.
"""

SAMPLE_TEXTS = {
    "News report (rich in entities)": (
        "Dr. Ananya Sharma joined Infosys in Bengaluru on 12 March 2021 and now "
        "leads a research team of forty engineers. Last quarter the company "
        "invested about $4.5 million in a natural language processing lab near "
        "the Electronic City campus. According to Reuters, two more centres will "
        "open in Hyderabad and Pune before December 2026."
    ),
    "Academic abstract (dense noun phrases)": (
        "This study evaluates transformer-based architectures for low-resource "
        "machine translation. We fine-tuned three pretrained models on a parallel "
        "corpus of 84,000 sentence pairs and measured performance using BLEU and "
        "chrF. The results suggest that subword segmentation contributes more to "
        "translation quality than model depth alone."
    ),
    "Conversational text (many stopwords)": (
        "I was just wondering if you could tell me whether the train to Haridwar "
        "is still running, because I had been waiting at the station for almost "
        "two hours and nobody there seemed to know anything about it at all."
    ),
    "Inflected sentences (good for lemmatization)": (
        "The children were running faster than the mice that had escaped from the "
        "broken cages. She studies better when the leaves are falling and the "
        "geese have flown south."
    ),
}
