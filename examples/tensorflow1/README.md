# Classes for TensorFlow1
This folder contains an example showing how to use the first implementation of Classes developed as a tool for TensorFlow1

## How to use it
In order to use Classes with TensorFlow1 we have to create a session and restore the weights of a trained model 
```python
with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, "weights")
```
Having done so we can instantiate the error simulator, defined in [`src/error_simulator_tf1.py`](../../src/error_simulator_tf1.py), it takes the session as input.

```
 error_sim = simulator.FaultInjector(session)
```

When we have created the error simulator, we need to instantiate a feed dictionary whose keys are the input layers of the model. 

```python
feed_dict = {
                model[0]: input_img
            }
```
Then we invoke the error simulator's `instrument` function, which takes two inputs. The first is a list of layers for which we want to obtain the output tensor. The second is the previously instantiated feed dictionary.

```python
error_sim.instrument([model[-1]], feed_dict)
```

This function will generate an internal copy of the session with the simulator. Then we must invoke the `generate injection sites` function to create a list of injectable sites. Finally, we can call the `inject` function. It takes the same inputs as the `instrument` one and returns the corrupted outputs of each layer requested.

```python
for _ in range(repetitions):
    res = error_sim.generate_injection_sites('OPERATOR_SPECIFIC', 1, instance[0],
    op_instance=instance[1])
    results, _, _ = error_sim.inject(
        [model[3],feed_dict)
```

---
We provide this documentation and the example [`cifar10.py`](cifar10.py) for completeness but this version of the framework is **NOT** mantained.