## Normal Models
|News Encoder|User Encoder|enable fields|sequence length|history length|hidden dim|batch size|learning rate|AUC|MRR|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|1DCNN|GRU|title|32|50|300|64|1e-4|0.6826|0.3292|
|Bert|GRU|title|32|50|768|64|1e-5|0.7126|0.3524|
<!-- |Bert|GRU|abstract|64|50|768|64|1e-5|0.7126|0.3524|
|Bert|GRU|title, abstract|96|50|768|64|1e-5|0.7126|0.3524| -->

## GateFormer
|News Encoder|User Encoder|Weighter|enable fields|sequence length|k|history length|gate hidden dim|batch size|learning rate|AUC|MRR|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Bert|GRU|CNN|title|32|4|64|768|64|1e-5|0.7158||
|Bert|GRU|CNN|title|32|4|64|300|64|1e-5|||
|Bert|GRU|CNN|title|32|4|100|768|64|1e-5|||
|Bert|GRU|CNN|title|32|4|64|768|100|6e-6|||
