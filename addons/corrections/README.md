# Add-on 4 — Box corrections (≈ 3–5 h)

500 **train-split** images with the zero-shot (J1) boxes as pre-labels. Fix them (move, resize, delete wrong boxes, add missing ones) so the detector is trained on human-corrected boxes (`detector_mode=HUMAN_CORRECTED`). Retrains the detector only.

Only these images may be corrected (they are all train split). Never put test images here.

Label Studio was **not installed** in the build environment, so these steps follow its documented Local Files workflow (written for Label Studio 1.x; check yours with `label-studio --version`). If a menu name differs in your version, the intent is: create a project, paste the labelling config, connect the folder as Local Files storage, import the tasks.

## Steps
1. Same Label Studio setup as the gold kit, but the storage path is `<repo>/addons/corrections/images` and the labelling config is `addons/corrections/label_config.xml`.
2. *Import* → upload `addons/corrections/import_with_prelabels.json` (the boxes arrive as *predictions*; accept or edit them per image).
3. Correct the boxes and Submit. Export with **YOLO with Images** (or *YOLO*) so you get `labels/*.txt` (class order = the order in the config: shikhara, vimana, gopuram, mandapa, pillar, jagati_base, amalaka).
4. Unzip so that the label files end up in `data/boxes/corrected/labels/<image_id>.txt`.
5. Run `claude --model sonnet "$(cat prompts/03_HUMAN_ADDONS.md)"`.
