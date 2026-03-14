# BSL Sign Language Recognition — Experiment

**Live demo: [chrisns.github.io/bsl-experiment](https://chrisns.github.io/bsl-experiment/)**

An experiment in browser-based British Sign Language recognition, built over 4 days using [NDX:Try](https://aws.try.ndx.digital.cabinet-office.gov.uk) sandbox AWS environments, Claude Code, and open academic datasets.

## What it does

Point your webcam at someone signing BSL. The browser extracts body/hand/face landmarks using MediaPipe, feeds 142-dimensional feature vectors into an ONNX neural network classifier, and displays the recognised sign in real time. No server required — everything runs client-side.

## How it was built

Nobody wrote any code. The entire project — from CloudFormation templates to PyTorch training pipelines to browser-side inference — was built using [Claude Code](https://docs.anthropic.com/en/docs/claude-code) driven by the [BMAD methodology](https://github.com/bmad-artifacts/bmad-methodology). A technical specification described the desired outcome; the AI did the implementation.

## Model versions

| Version | Signs | Accuracy | What changed |
|---------|-------|----------|-------------|
| v14 | 119 | 11.8% | ML with synthetic data (1 video per sign) |
| v15 | 119 | 86.6% | Real multi-signer data from BSLDict |
| v16 | 944 | 85.7% | 8x vocabulary expansion |
| v18 | 14,948 | 89.6% | 7 data sources, 27,000+ videos, 4-day EC2 training |
| v19 | 18,871 | 77.8% | GPU-trained (T4), largest vocabulary |

The deployed model (v19) runs as a 30MB ONNX file in the browser using ONNX Runtime Web.

## Data sources

Training data was sourced from publicly available academic sign language datasets:

- **BSLDict** (Oxford VGG) — 13,090 videos, 124 signers
- **BSL SignBank** (UCL) — 3,586 videos
- **Auslan Signbank** — 8,561 videos (Australian Sign Language, ~82% shared vocabulary with BSL)
- **NZSL** — 4,805 videos (New Zealand Sign Language)
- **Dicta-Sign** — 1,019 videos (EU research project)
- **SSC STEM** — 2,682 videos (Scottish Sensory Centre)
- **Christian-BSL** — 580 videos
- **BKS** — 2,072 videos

Video datasets are not included in this repository due to licensing constraints. See [docs/bsl-data-sources.md](docs/bsl-data-sources.md) for details.

## Repository structure

```
frontend/          Browser app deployed to GitHub Pages
cloudformation/    AWS deployment template (SageMaker, Bedrock, Lambda)
training/          ML training scripts (PyTorch → ONNX)
research/          Sign definitions, batch results, phonological data
docs/              Tech spec, data sources, screenshots
```

## Limitations

This is an experiment, not a product. See the [blog post](blog.md) for honest assessment:

- Vocabulary gaps — BSL has 20,000-100,000 signs in active use
- No BSL grammar — recognises isolated signs, not connected language
- Not tested with deaf BSL users
- Accuracy numbers are from reference videos, not real-world signing
- Licensing of training data is a patchwork

## Running locally

```bash
cd frontend
python3 -m http.server 8000
# Open http://localhost:8000
```

Requires a webcam and a modern browser with WebGL support.

## License

Code is open source. Training data has varied licensing — see [docs/bsl-data-sources.md](docs/bsl-data-sources.md).

## Background

This experiment was conducted as part of [NDX:Try](https://aws.try.ndx.digital.cabinet-office.gov.uk), which provides UK public sector organisations with free, temporary AWS sandbox environments for innovation and experimentation. The full AWS deployment (with SageMaker GPU inference, Bedrock Claude for text-to-BSL, and the complete infrastructure) can be deployed using the CloudFormation template in `cloudformation/`.
