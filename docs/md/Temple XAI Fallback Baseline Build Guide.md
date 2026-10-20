# Temple XAI Fallback Baseline: Build Guide

Oct 18, 2026 · @Ayush Aman

## Purpose and strategy

No open-source project does feature-aware, explainable Indian temple classification, so the fallback assembles one from pretrained models and one close methodological template. Each tier is a complete, demo-able system; later tiers add the research contribution.

| Tier | What it is | Annotation needed | Covers ablation | Risk |
|----|----|----|----|----|
| A | ImageNet-pretrained EfficientNet-B0 and MobileNetV3 fine-tuned on a temple-disjoint split, Grad-CAM heatmaps | Style label per temple only | Experiment A | Near zero |
| B | Tier A plus a zero-shot open-vocabulary detector (Grounding DINO or OWLv2) prompted with component names; detected boxes become component features fused with CNN features | None for training; ~150 hand-boxed test images for IoU | Experiments B and D | Low |
| C | Tier B with a YOLO detector fine-tuned on your own component boxes (auto-prelabelled by Tier B, then corrected), plus the IKS knowledge layer driving explanations | ~500 corrected images | Experiments C and D, the full paper | Medium |

Tier A alone already reproduces the 2026 baseline. Tier B is the minimum that makes the project's research claim testable. Tier C is the target.

### What already exists

| Source | What it gives you | Limits |
|----|----|----|
| [X-NeSyL / EXPLANet](https://github.com/julessanchez/x-nesyl) (MonuMAI) | The closest template: a part-based classifier over detected architectural elements, an expert knowledge graph, SHAP-guided training and a SHAP graph-edit-distance alignment metric. Code and trained models are published | European facades, not temples; older PyTorch and RetinaNet code |
| [Somala lab study, 2026](https://www.researchgate.net/lab/Surendra-Nadh-Somala-Lab-5) | Your direct baseline: 9,046 images, 218 temples, three classes, five lightweight CNNs, Grad-CAM | Dataset is custom; ask the authors whether it can be shared |
| [3DITA](https://doi.org/10.5281/zenodo.17295322) | Component vocabulary for Nagara temples (walls, mandapa, shikhara, amalaka, garbhagriha) | Point clouds, available on request; useful for the 3D extension only |
| [indian_landmark_recognition](https://github.com/AKASH2907/indian_landmark_recognition) | Older Keras saliency + ImageNet-feature style classifier | 2018 code, random split; cite as prior work |
| [South Indian Temple Monuments (Kaggle)](https://www.kaggle.com/datasets/likhithd007/capstone-organized-data) | ~20k Dravidian-leaning images | Built for inpainting; provenance and licences unclear, so not for your main dataset |

The X-NeSyL paper is the key reference: it already proposes classifying a monument by its detected parts and measuring how far the model's SHAP attributions differ from an expert knowledge graph. Your contribution is that pipeline for Indian temples, with temple-disjoint evaluation and Grad-CAM-to-component IoU.

## Architecture

Two swappable interfaces carry the whole fallback: StyleModel for classifiers and ComponentDetector for detectors. Moving from Tier B to Tier C changes one config line.

<img src="media/image1.png" style="width:6in;height:5.80357in" alt="fallback pipeline, two swappable interfaces" />

fallback pipeline, two swappable interfaces

The fusion classifier alone sets the tradition. Grad-CAM is scored against human gold boxes, SHAP explains the component features, and the taxonomy supplies sourced definitions; none of them can change the prediction.

### Folder layout

templexai/\
AGENTS.md CREDITS.md\
templexai/\
device.py seed.py\
data/ wikidata_temples.py, commons_images.py, make_splits.py\
models/ baselines.py, fusion.py\
detect/ zeroshot.py, yolo.py\
xai/ gradcam.py, shap_components.py\
knowledge/ taxonomy.yaml, explain.py, llm_phrase.py (optional)\
eval/ metrics.py, explain_metrics.py, mcnemar.py\
api/ main.py\
configs/ data/ runs/ artifacts/ experiments/ notebooks/\
web/ Vite + React + TS + Tailwind\
tests/ docs/ third_party/

## Data sources

No public dataset has both temple style labels and component boxes, so you build one from Wikidata and Wikimedia Commons, which give you temple identity, location and per-image licences programmatically. Nothing on Roboflow Universe covers Indian temple components.

The pipeline, which the agent automates in P1:

1.  **Temple list from Wikidata.** Query Hindu temples located in India with their Commons category, coordinates, state and, where present, architectural style. This gives you a temple_id for every image, which the temple-disjoint split depends on.

2.  **Operational taxonomy (Member 1, by hand).** Assign each temple one of: Nagara, Dravida, Deccan / traditionally classified Vesara. Record the source for every label; temples without a sourced label are excluded, not guessed.

3.  **Images from Commons.** For each temple's category, pull images through the Commons API with licence and author metadata. Keep public domain, CC0, CC BY and CC BY-SA only. Cap at about 40 images per temple and drop near-duplicates by perceptual hash.

4.  **Image type tag.** Mark each image exterior-full, exterior-partial, detail or interior with CLIP zero-shot, then spot-check. Train on exterior images; keep the rest for robustness tests.

5.  **Split by temple.** 70 / 15 / 15 by temple with a stratified group split and a fixed seed, written once to splits.csv and never regenerated.

Target size: about 150 to 250 temples and 4,000 to 8,000 images, comparable to the 2026 baseline's 218 temples.

### Component annotations

Classes: shikhara, vimana, gopuram, mandapa, pillar, jagati / base, amalaka. Two sets, kept separate:

| Set | Size | Made how | Used for |
|----|----|----|----|
| Gold test boxes | 150 to 300 test-split images | Drawn by hand, checked by Member 1 | Explanation IoU, pointing game, detector mAP |
| Training boxes (Tier C) | About 500 train-split images | Grounding DINO pre-labels, corrected in Label Studio | Fine-tuning the YOLO detector |

Never let pre-labels into the gold set uncorrected; that would make the explanation metrics measure Grounding DINO instead of your model.

## Download checklist

Everything runs on a Mac (PyTorch's MPS backend) except detector fine-tuning and bulk Grounding DINO pre-labelling, which go on a free Colab or Kaggle GPU notebook. The code picks CUDA, then MPS, then CPU automatically.

### Software

- [ ] Miniconda with a Python 3.11 environment named templexai

- [ ] PyTorch and torchvision (MPS build on Mac; Colab already has CUDA)

- [ ] pip install timm transformers accelerate grad-cam captum shap scikit-learn pandas numpy opencv-python pillow imagehash requests SPARQLWrapper pyyaml matplotlib pytest

- [ ] pip install ultralytics (Tier C detector)

- [ ] pip install fastapi uvicorn python-multipart pydantic (backend)

- [ ] pip install label-studio (annotation tool, runs locally in the browser)

- [ ] Node.js 20 LTS (React + TypeScript + Tailwind frontend via Vite)

- [ ] A Google account for Colab or a Kaggle account for GPU notebooks

- [ ] Optional: Docker Desktop for deployment, Neo4j Desktop only if you build the knowledge graph extension

### Pretrained models (pre-cache once with a script so demos work offline)

- [ ] efficientnet_b0 and mobilenetv3_large_100 ImageNet weights via timm

- [ ] IDEA-Research/grounding-dino-tiny (and -base if the GPU allows) via Hugging Face

- [ ] google/owlv2-base-patch16-ensemble as the second zero-shot detector

- [ ] openai/clip-vit-base-patch32 for image-type tagging and a zero-shot style baseline

- [ ] The current small YOLO detection checkpoint from Ultralytics (Tier C starting weights)

### Repos and papers to keep in third_party/ and docs/refs/

- [ ] julessanchez/x-nesyl: EXPLANet part-based classifier, knowledge-graph alignment, SHAP-GED metric

- [ ] jacobgil/pytorch-grad-cam: examples and its CAM metrics module

- [ ] AKASH2907/indian_landmark_recognition: prior Indian-architecture classifier, for related work

- [ ] X-NeSyL paper (arXiv 2104.11914), the 2026 temple-style paper, and the 3DITA paper as PDFs

### Data

- [ ] Nothing to download by hand: P1's script pulls Wikidata and Commons

- [ ] Optional: email the 2026 study's authors about dataset access, and the 3DITA authors if you attempt the 3D extension

Freeze versions after installing (pip freeze \> requirements.lock) so the agent never drifts onto a different transformers or ultralytics API.

## Pick list

Pretrained models and libraries are used as dependencies; X-NeSyL is mined for ideas and evaluation design, not copied wholesale. The fusion model, explanation evaluation and knowledge layer are written fresh.

| Source | Take | Ignore | Feeds module |
|----|----|----|----|
| timm | efficientnet_b0, mobilenetv3_large_100 backbones, create_model(..., num_classes=3), feature extraction with forward_features | Its training scripts (write a short loop instead) | Tier A baselines, fusion backbone |
| pytorch-grad-cam | GradCAM, GradCAMPlusPlus, EigenCAM; its metrics.road / ROAD faithfulness tests | Its ViT reshape helpers unless you add a ViT | Explainability |
| Captum / SHAP | Integrated Gradients and occlusion for images; SHAP KernelExplainer or TreeExplainer over the component-feature vector | DeepSHAP on the whole CNN (slow, fragile) | Feature attribution |
| Grounding DINO, OWLv2 (Hugging Face) | Zero-shot boxes from text prompts such as "temple tower", "gateway tower", "pillared hall", "stone pillar", "raised platform" | Their training code | Tier B detector, pre-labelling |
| Ultralytics YOLO | Fine-tuning on your corrected boxes, built-in mAP evaluation | Tracking, segmentation heads (unless you add masks later) | Tier C detector |
| CLIP | Zero-shot image-type tags; a zero-shot style baseline for the paper's comparison table | Fine-tuning CLIP | Data pipeline, extra baseline |
| X-NeSyL / EXPLANet | Part-based classifier design (detected parts to class), expert knowledge-graph format, SHAP-GED alignment metric idea | Its RetinaNet code and MonuMAI loaders | Fusion model, knowledge layer, evaluation |
| 2026 temple-style paper | Model list, training setup, McNemar test for comparing models | Its random architecture choices beyond B0 and V3 | Baselines, statistics |
| Wikidata + Commons APIs | Temple identity, location, Commons category, per-image licence and author | Wikipedia article text as labels | Dataset |

Why SHAP on component features rather than on pixels: the component vector is small and named (shikhara present, its area, its position), so SHAP values read directly as "the curved tower raised Nagara probability by X", which is the explanation the project promises.

## Build order and gates

The fallback follows your own 14-week plan: Tier A results by week 5, a demo-able Tier B by week 8, and the full system by week 12, leaving two weeks for the paper.

<img src="media/image2.png" style="width:6in;height:4.25893in" alt="build schedule over the 14-week plan, 3 gates" />

build schedule over the 14-week plan, 3 gates

Gate criteria: Tier A results means both baselines have test metrics on the temple-disjoint split. Tier B demo-able means the fusion model and component boxes show in a script or notebook. Full demo means every demo-flow box in the checklist is ticked. If week 8 slips, present Tier A plus zero-shot boxes; that still answers the research question honestly.

## Running the coding agent

Save the master prompt as AGENTS.md (or CLAUDE.md for Claude Code) in the repo root, put the project PDF at docs/project.pdf, then paste one module prompt per session.

1.  Create the repo, add AGENTS.md and docs/project.pdf, clone the references into third_party/.

2.  Paste the next module prompt and let it run to its report.

3.  Check three things: tests pass, the demo command works, git diff stays inside the module's folders. Commit.

4.  GPU-heavy steps (P3 pre-labelling, P6 YOLO training) produce a Colab notebook in notebooks/; open it in Colab, run all, download the outputs into artifacts/.

5.  If the agent fails twice on the same thing, paste the matching row from Troubleshooting.

Two modules need a human: Member 1 assigns style labels in P1, and the team draws gold boxes in P3. The prompts stop and wait at those points.

### Master context prompt (AGENTS.md)

You are building the FALLBACK BASELINE of an IKS research project:\
"Explainable Multimodal AI for Recognition and Interpretation of Indian Temple\
Architectural Heritage" (docs/project.pdf). Goal: a working, honest research\
system even if the team's custom models are unfinished. Reuse pretrained\
models and libraries; credit them.\
\
ARCHITECTURE (do not change without asking):\
\- Python 3.11 package \`templexai/\`: data/, models/, detect/, xai/,\
knowledge/, eval/, api/. Frontend in web/ (Vite + React + TS + Tailwind).\
\- Device: templexai/device.py picks cuda, then mps, then cpu.\
\- Every classifier implements StyleModel: predict(img) -\> probs,\
features(img) -\> tensor, target_layer for Grad-CAM.\
\- Every detector implements ComponentDetector: detect(img) -\> list of\
{label, box_xyxy, score}. Implementations: zeroshot (Grounding DINO /\
OWLv2) and yolo. Selected by config, never imported directly elsewhere.\
\- Component vocabulary and IKS knowledge live ONLY in knowledge/taxonomy.yaml.\
\- Data lives in data/: temples.csv, images.csv, splits.csv, boxes/.\
\
NON-NEGOTIABLE RESEARCH RULES:\
1\. Temple-disjoint splits from data/splits.csv only. Never split by image.\
Every training script asserts no temple_id overlap before training.\
2\. Never invent results. Numbers in reports come from files the code wrote.\
3\. The LLM (if any) only rephrases structured evidence; it never decides the\
class. The class comes from the vision pipeline.\
4\. Style labels come from data/temples.csv (human-assigned, sourced). Do\
not infer or edit labels.\
5\. Keep image provenance: every image row has source URL, licence, author.\
Drop images whose licence is not PD, CC0, CC BY or CC BY-SA.\
6\. Gold test boxes are never generated by a model.\
\
ENGINEERING RULES:\
7\. Work only in the folders the task names. Never refactor others.\
8\. Read the INSTALLED version's docs/source for timm, transformers,\
ultralytics, pytorch-grad-cam and captum before using their APIs.\
9\. Seed everything; configs in configs/\*.yaml; runs write to\
runs/\<name\>/\<timestamp\>/ with config, metrics.json, predictions.csv.\
10\. Each module ships pytest tests and a one-line demo command.\
11\. Code adapted from third_party/ gets a comment with URL and licence, plus a\
line in CREDITS.md.\
12\. End each task with a report: files changed, tests and results, demo\
command, known issues. Then stop.

## Module prompts

Paste these one per session, in order. Each assumes AGENTS.md is in place and earlier modules are committed.

### P0. Scaffold

Task P0: create the repo skeleton and shared contracts.\
\- Package templexai/ with data/, models/, detect/, xai/, knowledge/, eval/,\
api/; plus configs/, data/, runs/, artifacts/, notebooks/, tests/, web/,\
third_party/ (gitignored), docs/.\
\- templexai/device.py (cuda \> mps \> cpu), templexai/seed.py.\
\- Protocols: StyleModel and ComponentDetector exactly as AGENTS.md says.\
\- knowledge/taxonomy.yaml skeleton with the 7 components (shikhara, vimana,\
gopuram, mandapa, pillar, jagati_base, amalaka) and the 3 traditions,\
each with empty fields: display_name, definition, visual_cues, sources.\
\- data schemas (pydantic) for temples.csv, images.csv, splits.csv matching\
the metadata fields in docs/project.pdf section 13.\
\- CREDITS.md, requirements.txt, pytest config.\
Tests: schema validation, device selection.

### P1. Dataset builder

Task P1: templexai/data/ scripts that build the dataset reproducibly.\
1\. wikidata_temples.py: SPARQL for Hindu temples located in India with\
label, coordinates, state, Commons category, architectural style if\
present, heritage designation if present -\> data/temples_raw.csv. Look up\
the correct item and property IDs on Wikidata; do not guess them.\
2\. STOP and print instructions: Member 1 fills \`tradition\` (Nagara \|\
Dravida \| Deccan_Vesara \| exclude) and \`label_source\` in\
data/temples.csv. Provide a helper that pre-fills suggestions from the\
Wikidata style field, clearly marked as suggestions.\
3\. commons_images.py: for labelled temples, fetch images from the Commons\
category via the MediaWiki API with extmetadata (licence, author, URL);\
keep PD/CC0/CC BY/CC BY-SA; max 40 per temple; resize longest side to\
1024; dedupe with imagehash phash (distance \<= 6) -\> data/images/ and\
data/images.csv.\
4\. tag_image_type.py: CLIP zero-shot tags exterior_full, exterior_partial,\
detail, interior; store with score.\
5\. make_splits.py: StratifiedGroupKFold-based 70/15/15 by temple_id, seed\
42 -\> data/splits.csv. Refuse to overwrite an existing splits.csv.\
6\. dataset_report.py: counts per tradition, per state, per split, licence\
breakdown -\> docs/dataset_report.md.\
Tests: no temple in two splits; every image has licence and source.

### P2. Tier A baselines

Task P2: templexai/models/ and scripts/train_baseline.py.\
\- timm efficientnet_b0 and mobilenetv3_large_100, ImageNet weights,\
num_classes=3, exterior images only, temple-disjoint splits.\
\- Augmentation: random resized crop, flip, colour jitter. Class-balanced\
sampling or weighted loss. Early stopping on val macro F1.\
\- Outputs per run: metrics.json (accuracy, macro and weighted P/R/F1,\
per-class), confusion matrix PNG, predictions.csv, best.pt.\
\- scripts/clip_zeroshot.py: CLIP zero-shot style baseline on the test split.\
\- eval/mcnemar.py: paired McNemar test between any two predictions.csv.\
\- xai/gradcam.py: Grad-CAM and Grad-CAM++ via pytorch-grad-cam for any\
StyleModel, saving heatmap overlays for the test split.\
Runs on MPS on a Mac; if too slow, generate notebooks/train_baseline.ipynb\
for Colab that clones the repo and runs the same script.

### P3. Zero-shot components and annotation

Task P3: templexai/detect/zeroshot.py and the annotation workflow.\
\- ComponentDetector using Grounding DINO (IDEA-Research/grounding-dino-tiny)\
with OWLv2 (google/owlv2-base-patch16-ensemble) as an option. Prompt set\
per component defined in knowledge/taxonomy.yaml (several English\
phrasings each), per-class score thresholds in configs/detect.yaml.\
\- scripts/prelabel.py: run on chosen train images, export Label Studio\
pre-annotation JSON. notebooks/prelabel.ipynb for Colab GPU.\
\- scripts/labelstudio_setup.md: exact steps to import images +\
pre-labels, the labeling config XML for the 7 classes, and export to\
YOLO and COCO formats into data/boxes/.\
\- Gold set: select 200 test-split exterior images stratified by tradition,\
export them to Label Studio WITHOUT pre-labels. STOP and tell the team to\
draw gold boxes.\
Tests: detector output format; gold export contains no model boxes.

### P4. Feature-aware fusion model (Tier B)

Task P4: templexai/models/fusion.py.\
\- Component feature vector per image from any ComponentDetector: for each\
of the 7 classes: present (0/1), max score, count, total relative area,\
centroid y of largest box, aspect ratio of largest box (fixed order,\
documented in taxonomy.yaml).\
\- Model B: CNN pooled features (from the P2 backbone) concatenated with the\
component vector -\> 2-layer MLP. Option to freeze or fine-tune backbone.\
\- Model B-parts: components-only classifier (gradient boosting and a small\
MLP), the EXPLANet-style part-based classifier.\
\- Cache detector outputs to data/components\_\<detector\>.parquet so training\
never re-runs detection.\
Same metrics and outputs as P2. Report B vs A on the same test split.

### P5. Explanation evaluation

Task P5: templexai/xai/ and templexai/eval/explain_metrics.py.\
\- For each gold-boxed test image and each model:\
\* Pointing game: is the Grad-CAM max inside any gold box of a component\
that taxonomy.yaml lists as diagnostic for the predicted tradition?\
\* Energy-based pointing: share of CAM energy inside gold boxes.\
\* Explanation IoU: threshold CAM at top 10/20/30% pixels, IoU with the\
union of diagnostic gold boxes.\
\* Faithfulness: occlude (blur) the top component box and record the\
confidence drop; also ROAD from pytorch-grad-cam.\
\- SHAP on the P4 components-only model: per-image and global component\
importance, saved as plots and JSON.\
\- Background-reliance check: CAM energy outside all gold boxes.\
Output tables to runs/xai/\<ts\>/ for the paper.

### P6. Tier C trained detector

Task P6: templexai/detect/yolo.py and notebooks/train_yolo.ipynb.\
\- Train the current small Ultralytics YOLO detection model on the corrected\
boxes in data/boxes/ (train split only), validate on a held-out part of\
train, evaluate mAP50 and mAP50-95 on the gold test boxes.\
\- ComponentDetector wrapper; switch by configs/detect.yaml.\
\- Re-run P4 feature caching and P5 metrics with the YOLO detector.\
Compare zero-shot vs YOLO detector mAP on the gold set in one table.

### P7. IKS knowledge layer and explanations

Task P7: templexai/knowledge/.\
\- Fill-in tooling only: a validator for taxonomy.yaml (every component and\
tradition has definition, visual_cues, at least one source with URL or\
citation). Member 1 writes the content; do not write scholarly content\
yourself beyond placeholders marked TODO.\
\- explain.py: build a structured evidence object per prediction: predicted\
tradition + probability, detected components with boxes and scores,\
top SHAP contributions, Grad-CAM overlap per component, and the matching\
taxonomy entries with sources. Render it to a template-based report.\
\- Optional llm_phrase.py: rephrases the evidence object into prose; must\
quote only evidence fields and sources; the class field is read-only.\
\- Report must state the operational taxonomy caveat from project.pdf s11.

### P8. Ablation and robustness

Task P8: experiments/.\
\- Ablation A-D from project.pdf section 24, each over 3 seeds, same split:\
A baseline CNN, B CNN + components, C + knowledge-informed features or\
constraints, D full model with evidence alignment. Mean and std.\
\- Robustness on the test split: brightness/contrast shifts, random crops,\
synthetic occlusion patches, Gaussian blur and downscaling, and\
background masking (grey everything outside detected boxes).\
\- Output CSV tables and plots to experiments/results/, plus a markdown\
summary that only quotes those files.

### P9. Backend

Task P9: templexai/api/ with FastAPI.\
\- POST /analyze (image upload) -\> prediction, probabilities, components\
with boxes, Grad-CAM overlay (PNG base64), SHAP contributions, report\
text, provenance of any knowledge cited.\
\- GET /taxonomy, GET /health. Models load once at startup from artifacts/.\
\- Tests with FastAPI TestClient and a sample image.

### P10. Frontend

Task P10: web/ with Vite + React + TypeScript + Tailwind.\
\- Upload, then show: predicted tradition with confidence bar, image with\
toggleable component boxes, Grad-CAM overlay slider, component\
contribution bar list, the evidence report with source links, and a\
visible note that labels follow the project's operational taxonomy.\
\- A gallery of test-split examples (correct and incorrect) for the demo.

## Demo and testing checklist

The fallback is ready when every box is ticked on a Mac with Wi-Fi off. Run the list at week 9 and again before submission.

Research integrity

- [ ] splits.csv committed once; every training run logged its no-overlap assertion

- [ ] Every tradition label in temples.csv has a source; excluded temples listed with a reason

- [ ] Every image has source URL, licence and author; only PD, CC0, CC BY, CC BY-SA remain

- [ ] Gold boxes drawn by people, with no model pre-labels in that set

- [ ] Every number in the paper traces to a file in runs/ or experiments/results/

Models and metrics

- [ ] Tier A: B0 and V3 trained, test metrics and confusion matrices saved

- [ ] CLIP zero-shot baseline row in the comparison table

- [ ] Tier B fusion model compared with Tier A on the same split, with McNemar p-value

- [ ] Pointing game, explanation IoU and occlusion faithfulness reported for every model

- [ ] Detector mAP on the gold set for zero-shot and (if done) YOLO

- [ ] Ablation A to D over 3 seeds; robustness table

Demo flow

- [ ] uvicorn templexai.api.main:app and npm run dev start with cached models, no internet

- [ ] Upload a test-split image: tradition, confidence, boxes, Grad-CAM, contributions and report all appear

- [ ] Show one correct and one misclassified example and explain both from the evidence

- [ ] The operational-taxonomy caveat is visible in the UI and the report

- [ ] A screen recording of the demo exists as backup

## Credits and licences

Two licences need attention. Ultralytics YOLO is AGPL-3.0, which matters if you pursue the software-copyright route in section 34 of your plan; if so, train an Apache-licensed detector such as RT-DETR from Hugging Face instead. And CC BY / CC BY-SA images need author credit wherever they appear in the paper or UI.

| Source | Used for | Licence | Note |
|----|----|----|----|
| [timm](https://github.com/huggingface/pytorch-image-models) | Backbones | Apache-2.0 | ImageNet weights: research use |
| [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) | Grad-CAM, ROAD | MIT |  |
| [Captum](https://github.com/pytorch/captum) | Integrated Gradients, occlusion | BSD-3-Clause |  |
| [SHAP](https://github.com/shap/shap) | Component attribution | MIT |  |
| [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base) | Zero-shot detection, pre-labels | Apache-2.0 |  |
| [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) | Second zero-shot detector | Apache-2.0 |  |
| CLIP | Image-type tags, zero-shot baseline | MIT |  |
| Ultralytics YOLO | Tier C detector | AGPL-3.0 | Swap for RT-DETR if IP matters |
| [X-NeSyL](https://github.com/julessanchez/x-nesyl) | Part-based design, alignment metric | Check repo | Cite the Information Fusion paper |
| Label Studio | Annotation | Apache-2.0 |  |
| FastAPI, React, Tailwind | App | MIT |  |
| Wikidata | Temple metadata | CC0 |  |
| Wikimedia Commons | Images | Per image (PD, CC0, CC BY, CC BY-SA) | Keep author + licence per row; credit in figures |

Credit format for CREDITS.md and code headers: Adapted from \<URL\> (\<licence\>), \<what was taken\>. Image credits are generated automatically from images.csv into docs/image_credits.md.

## Troubleshooting

The usual failures are leakage between splits, weak zero-shot detections on temple-specific parts, and API drift. Paste the "tell the agent" text as-is.

| Symptom | Likely cause | Tell the agent |
|----|----|----|
| Test accuracy suspiciously high (95%+) | Temple leakage or near-duplicate images across splits | "Re-run the temple overlap assertion and a cross-split phash duplicate check; report every collision." |
| One tradition dominates predictions | Class imbalance by temple count | "Use class-weighted loss and per-class temple caps; report macro F1, not accuracy." |
| Grounding DINO misses shikhara or vimana | English prompts too vague for Indian terms | "Add descriptive prompts per component in taxonomy.yaml (e.g. 'curved stone temple tower', 'stepped pyramidal temple tower'), lower that class's threshold, report recall on 30 gold images." |
| Grad-CAM looks blank or uniform | Wrong target layer | "Print the backbone's last conv block name from the installed timm model and use that as target_layer." |
| MPS errors or very slow training | Unsupported op on Apple GPU | "Set PYTORCH_ENABLE_MPS_FALLBACK=1; if still slow, generate the Colab notebook for this step." |
| Commons API rate-limited | Too many fast requests | "Add a descriptive User-Agent, sleep between requests, and resume from a cache file." |
| transformers or ultralytics errors | Version drift | "Stop guessing; read the installed package source and examples, then match that API." |
| Fusion model no better than CNN | Component features too sparse or noisy | "Report it honestly as a finding; then try the YOLO detector and per-class score calibration." |
| Agent writes scholarly text into taxonomy.yaml | Overreach | "Revert. Only Member 1 writes taxonomy content; leave TODO placeholders." |
| Agent edits unrelated modules | Context drift | "Revert changes outside \<folders\>. Re-read AGENTS.md rule 7 and redo within scope." |
