# Deephack.RL qualification round solution

Based on coreylynch/async-rl, adapting tweaks from miyosuda/async-deep-learning

Implements standard A3C algorithm with entropy regularization.

To train:
```python main.py```

Weights checkpoints would appear every minute under 'checkpoints' directory. With training parameters specified in config.py, solution tends to converge quickly to exclusive use of a single action all the time. Which particular action, depends on random seed initialization. The checkpoint provided with solution, naturally reflects the case when it converged to using the NO-OP, thus successfully avoiding any action in skiing.

To test:
```python main.py --testing True --checkpoint_path checkpoints/checkpoint```

The provided weights ensure consistent -9013.0 score in skiing.

