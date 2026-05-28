# PostTraining with Zero Code

LLM post-training should not feel like babysitting 20 scripts, manually hunting datasets, checking logs at 3 a.m.🫸, and praying your eval score 🤗. Try this, post-training should be easy as ***t.

## Quick Start

> Fill one YAML. Run one command. Let the agent do the boring work.  

```sh
pip install -e .
autopilot-autonomous --config autopilot.yaml \
  --goal "Improve ../qwen3-1.7b on code generation tasks." \
  --output-dir runs/coding \
  --max-hours 10
```


## Demo

![visualize .PNG example](./example.PNG)

