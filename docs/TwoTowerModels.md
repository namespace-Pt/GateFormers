## Normal Models
|News Encoder|User Encoder|enable fields|sequence length|history length|hidden dim|batch size|learning rate|AUC|MRR|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|1DCNN|GRU|title|32|50|300|64|1e-4|0.6826|0.3292|
|Bert|GRU|title|32|50|768|64|1e-5|0.7126|0.3524|
<!-- |Bert|GRU|abstract|64|50|768|64|1e-5|0.7126|0.3524|
|Bert|GRU|title, abstract|96|50|768|64|1e-5|0.7126|0.3524| -->

## GateFormer
|News Encoder|User Encoder|Weighter|enable fields|sequence length|k|history length|hidden dim|batch size|learning rate|AUC|MRR|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Bert|GRU|CNN|title|32|4|50|768|64|1e-5|0.6899|0.2948|
|Bert|GRU|CNN|title|32|4|50|768|100|1e-5|0.6906|0.2921|
|Bert|GRU|CNN|title|32|4|50|768|64|3e-6|0.6863|0.2907|
|Bert|GRU|TFM|title|32|4|50|768|64|1e-5|0.6909|0.2908|
|Bert|GRU|First|title|32|4|50|768|64|1e-5|0.6859|0.2907|
|Bert|GRU|BM25|title|32|4|50|768|64|1e-5|0.6882|0.2874|
|Bert|GRU|CNN|title, abs|96|10|50|768|64|1e-5|0.7003|0.2948|
|Bert|GRU|First|title, abs|96|10|50|768|64|1e-5|||
|Bert|GRU|BM25|title, abs|96|10|50|768|64|1e-5|||
|Bert|GRU|KeyBert|title, abs|96|10|50|768|64|1e-5|||


|Bert|GRU|CNN|title|32|8|50|768|64|1e-5|0.7039|0.2994|
|Bert|GRU|First|title|32|8|50|768|64|1e-5|||
|Bert|GRU|BM25|title|32|8|50|768|64|1e-5|||
|Bert|GRU|KeyBert|title|32|8|50|768|64|1e-5|||

|k|gate|batch size|AUC|MRR|nDCG|
|:-:|:-:|:-:|:-:|:-:|
|1|First|||
|1|Cnn|||
|2|First|||
|2|Cnn|
|4|First|0.6914|0.293|0.3951|
|4|Cnn|||
|6|First|||
|6|Cnn|||
|8|First|0.7015|0.3458|0.4459|
|8|Cnn|||
