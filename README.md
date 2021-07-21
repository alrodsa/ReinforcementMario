## Abstract
Using ***Reinforcement Learning*** (RL), concretely ***Deep Q-Learning*** with *TensorFlow 2.0* and *Keras-RL* library to make Mario learn how to play in Super Mario World. 

## Dependencies installation
 | Library | Installation command |
 | ---- | ---- |
 |Gym Retro | ```pip3 install gym-retro``` |
 |Tensorflow 2.0| ```pip3 install tensorflow``` |
 |Keras-rl| ```pip install keras-rl``` |
 |OpenCV    | ```pip3 install opencv-python```|


[IMPORTANT] The ROM of *Super Mario World* must be imported to use it in the Gym Retro enviroment, import it with:

```python3 -m retro.import /path/to/your/ROMs/directory/```

## Update RAM variables in *data.json*
Open the file *data.json* located in Python libraries installation folder:

```../lib/python[3.6|3.7|3.8]/site-packages/retro/data/stable/SuperMarioWorld-Snes/```

and subsitute it with the following RAM addresses for the new variables:
```
{
  "info": {
    "checkpoint": {
      "address": 5070,
      "type": "|i1"
    },
    "coins": {
      "address": 8261055,
      "type": "|u1"
    },
    "endOfLevel": {
      "address": 8259846,
      "type": "|i1"
    },
    "lives": {
      "address": 8261399,
      "type": "|u1"
    },
    "powerups": {
      "address": 25,
      "type": "|i1"
    },
    "score": {
      "address": 8261428,
      "type": "<u4"
    },
    "x": {
      "address": 148,
      "type": "<u2"
    },
    "dead": {
      "address": 8257688,
      "type": "<u4"
    },
    "y": {
      "address": 114,
      "type": "<u4"
    },
    "jump": {
      "address": 8257747,
      "type": "<u4"
    },
    "yoshiCoins": {
      "address": 8262690,
      "type": "<u4"
    }
  }
}
```

## Plots obtained from execution
It has been obtained execution statistics for **2 different methods**:

 - *Metodo de Resta de valores constantes* (MR) -> The penalties consist on concrete values that decrease the reward. 
 - *Metodo de Porcentajes* (MP) -> The penalties consist on percentage values that decrease the reward.

[Episode reward]

<img src="https://github.com/alrodsa/ReinforcementMario/blob/main/graphics/rewEpisode.svg">

[Maximun reward]

<img src="https://github.com/alrodsa/ReinforcementMario/blob/main/graphics/rewMaxEpisodeRL.svg">

[Duration]

<img src="https://github.com/alrodsa/ReinforcementMario/blob/main/graphics/durationRL.svg">

[Mean reward]

<img src="https://github.com/alrodsa/ReinforcementMario/blob/main/graphics/rewMeanRL.svg">



