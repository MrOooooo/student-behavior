# Model 4: Open Model Zoo Integration

Model 4 adds the OpenVINO Open Model Zoo Smart Classroom model:

- Model: `person-detection-action-recognition-0006`
- Source project: `open_model_zoo`
- Runtime: OpenVINO
- Classes: `sitting`, `writing`, `raising_hand`, `standing`, `turned_around`, `lie_on_the_desk`

The model files are stored in:

```text
object_cheating/models/open_model_zoo/person-detection-action-recognition-0006/FP16/
```

Run the app as usual:

```bash
pip install -r requirements.txt
reflex run
```

Use the model navigation arrows until the page shows `Model 4`, then enable detection.

## Multi-Target Selection

The target selector now uses checkboxes. Within one active model, select any
combination of behavior classes to detect several behaviors at the same time.
Select `All` to disable class filtering for the active model.

## Cross-Model Detection

Detecting targets across models, such as Model 1 `Look Around` plus Model 4
`raising_hand`, does not require retraining if the existing models already
cover those classes.

Enable `Cross Model Detection` in the Detection Summary panel, then choose
target classes under each model group. The runtime will run the selected models
on the same frame and merge boxes, table rows, statistics, and saved crops.
