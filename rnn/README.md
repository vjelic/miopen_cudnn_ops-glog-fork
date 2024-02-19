
# Run RNN on cuDNN

Follow these steps:

1. Clone the repo and configure the correct `compute_#` and `sm_#` capability.

```bash
./setup.sh
```

2. Build the RNN sample.

```bash
./build.sh
```

3. Perform the RNN measurements.

```bash
./lstm_rnn.sh
```
