# brain_qa evaluation

- Questions: 15
- Extractive any-match accuracy: **93.3%** (14/15)
- Preferred-path in top-5: **86.7%**

| id | ok | path_ok | matched | question |
|----|----|---------|---------|----------|
| 1 | Y | Y | REAL, SHUF | what was real vs shuffled myelin NARMA? |
| 2 | N | Y |  | which myelin rule recovered most of real myelin's  |
| 3 | Y | Y | macaque | did macaque wiring beat random? |
| 4 | Y | N | XOR, EI, fly, CX | fly excitatory inhibitory XOR result |
| 5 | Y | N | DESIGN_EI_ONLY | what is DESIGN_EI_ONLY? |
| 6 | Y | Y | plasticity, p, 1 | plasticity experiment outcome |
| 7 | Y | Y | gated, Frankenstein | gated frankenstein specialist readout result |
| 8 | Y | Y | atlas, wiring, EBRAINS, no | EBRAINS macaque what data? |
| 9 | Y | Y | 0.4599, RULE_FF_DEPTH | RULE_FF_DEPTH NARMA mean |
| 10 | Y | Y | 0.393, REAL, NARMA | human myelin REAL NARMA from myelin_real_rep |
| 11 | Y | Y | celegans | celegans sources connectome |
| 12 | Y | Y | finch, song, HVC | finch songbird wiring experiment |
| 13 | Y | Y | Bonferroni, SHUF, RULE | Bonferroni primary for myelin mechanism |
| 14 | Y | Y | EI_SHUF, XOR | xor_cx EI_SHUF compared to CX |
| 15 | Y | Y | dopamine, serotonin, neuromod | neuromod real dopamine serotonin test |

Accuracy = fraction of questions where at least one expected token/number appears in the extractive answer or citation paths. Not generative EM. distilgpt2 generative head not used: too weak for reliable numeric answers; extractive default kept after protocol.
