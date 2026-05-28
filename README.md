# PostTraining with Zero Code

LLM post-training should not feel like babysitting 20 scripts, manually hunting datasets, checking logs at 3 a.m.🫸, and praying your eval score 🤗. Try this, post-training should be easy as ***t.

## Quick Start

> Fill the `autopilot.yaml`, including:

- LLM API_KEY, BASE_URL, MODEL_NAME;

- SERPER_API_KEY; 

- Installed training environments.

> Run this:

```sh
pip install -e .
autopilot-autonomous --config autopilot.yaml \
  --goal "Improve ../qwen3-1.7b on code generation tasks." \
  --output-dir runs/coding \
  --max-hours 5
```

> Go to bed.

## Example

![visualize .PNG example](./example.PNG)

