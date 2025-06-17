## Pretraining

clone:
```
git clone git@github.com:allenai/OLMo.git
git checkout DataDecide
cd OLMo
```

Setup only needs gantry:
```
conda create -n just_gantry python=3.10
conda activate just_gantry
pip install beaker-gantry
```

There are several hardcoded edits you will have to make!
Make sure to use your WandB creds in [this script](https://github.com/allenai/OLMo/blob/55d4871d7777f5cc4561f8e508f635f9c6308bbc/scripts/beaker/ladder-launch.sh#L38):
```
--env-secret WANDB_API_KEY=<your creds in beaker> \
```

Also [here](https://github.com/allenai/OLMo/blob/55d4871d7777f5cc4561f8e508f635f9c6308bbc/scripts/ladder.py#L141):
```
remote_save_folder = f"s3://ai2-llm/checkpoints/<insert path here>/{run_name}"
```

And [here](https://github.com/allenai/OLMo/blob/55d4871d7777f5cc4561f8e508f635f9c6308bbc/scripts/ladder.py#L188):
```
project=<insert a new wandb project name>
```

Now we generate a script with launch commands for all our experiments:
```
python create_ladder_over_scale_script.py --scales eval-for-consistent-ranking-scales.txt --data-mixes eval-for-consistent-ranking-mix-names.jsonl --length 5xC --seeds <seeds> > my_experiments.sh
```

Use this script from the root of your OLMo repo to launch all the experiments.

### Simple training example

For quick experiments you can run our lightweight training script with a YAML configuration:

```bash
python train_simple_olmo.py --config train_config.yaml
```

`train_simple_olmo.py` initializes an `OLMoConfig` with `embedding_size` set to the selected model's hidden dimension so the model parameters are well-defined.  Edit `train_config.yaml` to reference your dataset and desired parameters.  If `eval_tasks` are provided, the script evaluates the resulting checkpoint on those lm_eval tasks.
