# Bias Detector for a Pretrained Language Model

## Run locally
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:
   `pip install -r requirements.txt`
5. Start:
   `streamlit run app.py`

The default model is `distilgpt2`. The first run may download model weights. Use a machine with enough memory. Change the Hugging Face model ID in the sidebar to test another compatible text-generation model.

## What's included
- Editable prompt templates and reference sets
- Matched prompt generations with shared per-trial seed
- Side-by-side comparison and text similarity
- Simple transparent lexical flags (not validated bias measures)
- JSON report and CSV export

## Experimental controls / suggested protocol
- Keep template wording, model revision, decoding settings, and trial count fixed.
- Change only the reference variable intended for the test.
- Use multiple prompt templates and reference pairs, with repeated generations.
- Record exact model ID/revision, package versions, parameters, seed, and run date.
- Inspect completions manually. Consider blinded annotation by multiple reviewers and report agreement.
- If comparing names, do not assume a name maps reliably to a demographic identity.
- Paired seeds improve repeatability but do not guarantee identical stochastic conditions because tokenization and continuation contexts can differ.

## Limitations
- One template, pair, or output cannot establish that a model is biased.
- Lexical keyword flags are crude, context-blind proxies.
- Surface similarity is not semantic fairness.
- Outputs may vary with sampling, model revision, prompt wording, and decoding parameters.
- The tool is an exploratory audit, not a validated fairness certification.
