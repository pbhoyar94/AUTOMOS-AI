## Prepare env variable with correct path

- To let the project know where is map and experiment folder, you need to set environment vairables: ```NUPLAN_DATA_ROOT```, ```NUPLAN_DATA_ROOT``` and ```NUPLAN_DATA_ROOT```

First, we need to know the parent_dir of the downloaded NuPlan dataset, in the README example:
```bash
${HOME}/nuplan
```

The ```parent_dir``` is ```${HOME}```. In case your ```parent_dir``` is something else, just replace the path with ```parent_dir``` below:

```bash
export NUPLAN_DATA_ROOT="parent_dir/nuplan/dataset"
export NUPLAN_MAPS_ROOT="parent_dir/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="parent_dir/nuplan/exp"
```

For example, you can have    
```bash
export NUPLAN_DATA_ROOT="$HOME/nuplan/dataset"
export NUPLAN_MAPS_ROOT="$HOME/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="$HOME/nuplan/exp"
   ```

## Set env variable
1. Open your shell configuration file in ```nano```.
   ```bash
   nano ~/.bashrc
      ```

2. Add the following lines (according to the example) at the end of the file:
   ```bash
   export NUPLAN_DATA_ROOT="$HOME/nuplan/dataset"
   export NUPLAN_MAPS_ROOT="$HOME/nuplan/dataset/maps"
   export NUPLAN_EXP_ROOT="$HOME/nuplan/exp"
      ```
Save the changes by pressing ```Ctrl + O```, then press Enter to confirm the filename. Exit nano by pressing ```Ctrl + X```.

3. Apply the changes by reloading the configuration file:
      ```bash
      source ~/.bashrc
      ```