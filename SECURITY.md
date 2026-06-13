# Security policy

## Credentials

Never commit API keys, model-hub tokens, bank or university credentials, or populated `.env` files.
Use `.env.example` only as a variable-name template. If a secret is committed or shared, revoke it
at the provider immediately; deleting the file does not invalidate the credential or erase Git
history.

## Data and model artifacts

This public repository intentionally excludes private RAG corpora, Chroma databases, LoRA weights,
training checkpoints, and raw eCADSTAR projects/results. Confirm ownership and redistribution rights
before sharing any CAD file, document corpus, model adapter, or screenshot.

## Reporting

Report a suspected credential or data exposure privately to
[kovarthanmuthusamy@gmail.com](mailto:kovarthanmuthusamy@gmail.com). Do not open a public issue that
contains a secret or private engineering file.
