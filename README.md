1. 简化正文
2. 压缩
   1. 方法论1页
      - non-personalized
   2. 实验1页
3. 实验
  - 重构代码
  - transformer selection encoder
  - bert selection encoder
# This branch is MY WORK
Since the paper is under review, I cannot release more details.

## Environment
```
python=3.8.11
torch==1.9.1
```

## Instruction
```bash
# train
python tesrec.py

# test
python tesrec.py -m test
```

## TODO
|non-bert lr|bert lr|batch size|AUC|
|:-:|:-:|:-:|:-:|
|1e-4|3e-6|64||
|3e-6|3e-6|64||
|1e-5|1e-5|64||
|1e-5|1e-5|128||
|1e-5|1e-5|256||
|1e-5|1e-5|512||

- compare against paper [A Self-Attentive Model with Gate Mechanism for Spoken Language Understanding]
  - same: both optimize the learning of gating weight by the final NLL loss
  - diff: we learn a single gating weight for each vector, while this paper learns a gating vector of the same size as the input vector; gating in this paper cannot be used to filter out vector inputs, it can be viewed as an attention.

