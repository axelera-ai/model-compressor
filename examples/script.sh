for model_name in Routing ResNet18 ShuffleNetV2 U-Net SmolLM2-135M SigLIP
do
  for hardware_generation in alpha
  do
    for policy in z y x w
    do
      for coalescence_mode in without with
      do
        for assignment_mode in strict permissive
        do
          echo ${model_name} ${hardware_generation} ${policy} ${assignment_mode} ${coalescence_mode}
          python examples/flow.py --model-name ${model_name} --hardware-generation ${hardware_generation} --policy ${policy} --assignment-mode ${assignment_mode} --coalescence-mode ${coalescence_mode}
        done
      done
    done
  done
done
