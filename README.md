# edenv

`edenv` has `EditDistanceEnvironment`, a label-free way to generate PrimeRL environments for tasks that operate on the surface structure of text.

## Use

Import `EditDistanceEnvironment` and create an instance by supplying:
- A prompt template string
- A function for taking an input sample and generating the target string
- A task name
- One or both of:
  - A set of default samples
  - A huggingface dataset id and the column you want to use from it
    - Also may be specified at runtime with the Prime Env `-a` flag

See the example provided for a complete implementation.
