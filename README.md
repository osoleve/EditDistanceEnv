# edenv

`edenv` has `EditDistanceEnvironment`, a label-free way to generate PrimeRL environments for tasks that operate on the surface structure of text.

## Deps

- jellyfish
- datasets
- verifiers

### Example Deps

- nltk

## Use

Copy the `edenv.py` file into your env to get `EditDistanceEnvironment`. 

Create an instance by supplying:
- A prompt template string
- A function for taking an input sample and generating the target string
- A task name
- One or both of:
  - A set of default samples
  - A huggingface dataset id and the column you want to use from it
    - Also may be specified at runtime with the Prime Env `-a` flag
   
Connect the environment to PrimeEnv by assigning the `.load_environment` attribute of your env to the top level variable `load_environment`

See the example provided for a complete implementation.

## TODO

- It's clunky, should be importable.
- Could probably hook into the globals to set the top level variable automatically
- Add other edit distance metrics?
