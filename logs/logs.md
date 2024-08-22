# Folder Structure
```
logs
  |-<model_name>
  |       |-evaluations.npz (stats recorded during training)
  |       |-final.zip (model saved after 4M timesteps of training)
  |       |-best_model_<some_metric>_<eval_counter> (saved by evaluation callback)
  |
  |-<map_name>
        |-<model_name>_(<additional_info>)
                |-performance.py (model's evaluation performance)
                |-record_<id>.npz (various data recorded during evaluation)
```

# Naming
Naming conversion has changed quite a bit throughout the project.

Earlier models are named `<input>_<learning_rate>`:
- rgb_9e-4
- ss_1e-3
- ss_rgb_1e-3
- etc.

Later models are named `s<frame_stack>_<model_type>_<input>_<learning_rate>` or `r<frame_stack>_<recurrent_model_type>_<experiment_name>_<input>_<learning_rate>`:
- s4_ppo_ss_1e-3
- r1_r_ppo_Dsl_ss_5e-4
- etc.
