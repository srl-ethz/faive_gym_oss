overwrite this text to describe your changes here

## checklist
PR can be merged after all these are met
- [ ] describe the changes (with screenshots if it helps)
- [ ] If this PR modifies any part of the training, post the W&B results of the following experiments (post screenshot of the consecutive_successes)
    ```bash
    python train.py task=FaiveHandP0 capture_video=True force_render=False wandb_activate=True wandb_group=srl_ethz wandb_project=faive_hand wandb_name=faivehandp0_check
    ```